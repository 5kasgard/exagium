```hypothesis
Family: label-embmlp (per-position byte embeddings + one ReLU hidden layer).
Why: The task is character-level language modeling with 16-byte context. Linear models (label-linear) ignore interactions between positions. An embedding layer compresses each byte into a dense vector, and a hidden layer captures non-linear interactions. This factorized architecture has fewer parameters than a full one-hot MLP (label-mlp16) and should generalize better on this small dataset.
Failure: If dev accuracy <= label-linear (0.3375) or instructions/item exceed label-linear by >2x, the factorization is too restrictive or optimization fails.
```

```python
"""label-embmlp: per-position byte embeddings summed, one ReLU hidden layer, then output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D424D, int32 d,E,H,C; float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding size
H = 64          # hidden size
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Sum embeddings: (B, E)
            summed = np.zeros((B, E), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j]]   # (B, E)
            hidden = np.maximum(summed @ W1 + b1, 0)   # (B, H)
            logits = hidden @ W2 + b2                  # (B, C)

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                     # (B, C)

            # Backward
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T            # (B, H)
            grad_hidden[hidden == 0] = 0.0              # ReLU gradient
            grad_W1 = summed.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_summed = grad_hidden @ W1.T            # (B, E)

            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, scatter-add grad_summed to grad_emb[j, xb[:, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_summed)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424D) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    float *emb = (float *)(wb + 20);
    float *W1 = emb + (long)d * 256 * E;
    float *b1 = W1 + (long)E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // zero summed
        for (int e = 0; e < E; e++) summed[e] = 0.0f;
        // accumulate embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) summed[e] += eptr[e];
        }
        // hidden = ReLU(summed @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int e = 0; e < E; e++) acc += summed[e] * W1[(long)e * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }
        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
            logits[c] = acc;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```