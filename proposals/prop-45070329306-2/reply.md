```hypothesis
Propose label-embmlp: per-position byte embeddings summed, then a ReLU hidden layer.
Captures positional non-linear interactions missed by linear models; 32-dim embeddings + 128 hidden units
should exceed label-linear's 0.3375 dev accuracy. If dev accuracy ≤ 0.3375, the extra capacity does not help.
```

```python
"""label-embmlp: per-position embeddings + one ReLU hidden layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D424D  # 'EMBM' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: [d, 256, E]
    emb = (rng.normal(0, 1/math.sqrt(E), size=(d, 256, E))).astype(np.float32)
    # hidden layer: [E, H]
    W1 = (rng.normal(0, math.sqrt(2.0/E), size=(E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer: [H, C]
    W2 = (rng.normal(0, math.sqrt(2.0/H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = len(yb)

            # Forward
            # Embedding lookup and sum: [B, E]
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]   # broadcasting

            hidden_pre = emb_sum @ W1 + b1    # [B, H]
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2         # [B, C]

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                        # [B, C] gradient w.r.t logits

            # Backprop
            # Output layer
            grad_W2 = hidden.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_hidden = probs @ W2.T        # [B, H]
            grad_hidden[hidden_pre <= 0] = 0  # ReLU derivative

            # Hidden layer
            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb_sum = grad_hidden @ W1.T  # [B, E]

            # Embeddings: scatter-add per position
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, accumulate gradients for the byte values seen
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * (g * g)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Save weights
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        for p in params:
            f.write(p.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size: header(20) + emb(d*256*E) + W1(E*H) + b1(H) + W2(H*C) + b2(C) all float32
    long expected = 20 + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero embedding sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;

        // Accumulate embeddings for each position
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // Hidden layer: ReLU(emb_sum * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```