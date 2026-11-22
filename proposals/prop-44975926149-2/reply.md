```hypothesis
Family: label-embed. Learns 16 position-specific byte embedding tables (256×D), sums them, then passes through a single ReLU hidden layer (size H) to 65 outputs. Captures byte interactions with low inference cost (O(D·H + H·C)). With D=32, H=64 it should exceed label-linear's 33.7% dev accuracy; if it does not, the hypothesis fails.
```

```python
"""label-embed: position-specific byte embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 32          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D4244  # "EMBD" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: 16 tables of 256 x D
    emb = (rng.normal(0, 1.0/math.sqrt(D), (d, 256, D))).astype(np.float32)
    # Hidden layer
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / (D + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = EPOCHS * steps_per_epoch

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, 16)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Sum embeddings: (B, D)
            summed = np.zeros((B, D), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j]]   # (B, D)
            # Hidden layer
            hidden = summed @ W1 + b1        # (B, H)
            np.maximum(hidden, 0, out=hidden)  # ReLU
            # Output
            logits = hidden @ W2 + b2        # (B, C)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_out = probs / B             # (B, C)

            # Backprop
            # Output layer
            grad_W2 = hidden.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_hidden = grad_out @ W2.T    # (B, H)
            # ReLU
            grad_hidden[hidden == 0] = 0.0
            # Hidden layer
            grad_W1 = summed.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_summed = grad_hidden @ W1.T  # (B, D)
            # Embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, scatter-add grad_summed to grad_emb[j, xb[:, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_summed)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * (g * g)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Write weights.bin
    # Header: magic, d, D, H, C (5 int32)
    header = struct.pack("<5i", MAGIC, d, D, H, C)
    # Embeddings: d * 256 * D floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W1, b1, W2, b2
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(emb_bytes)
        f.write(W1_bytes)
        f.write(b1_bytes)
        f.write(W2_bytes)
        f.write(b2_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = (float *)malloc(sizeof(float) * D);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int j = 0; j < D; j++) summed[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) summed[j] += e[j];
        }

        // Hidden layer with ReLU
        for (int u = 0; u < H; u++) {
            float val = b1[u];
            for (int j = 0; j < D; j++) val += summed[j] * W1[(long)j * H + u];
            hidden[u] = val > 0.0f ? val : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float val = b2[c];
            for (int u = 0; u < H; u++) val += hidden[u] * W2[(long)u * C + c];
            scores[c] = val;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```