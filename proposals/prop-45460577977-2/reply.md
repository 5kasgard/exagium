```hypothesis
We propose a factorized linear model (label-factorized) that learns a low-dimensional embedding for each of the 16 context positions and a linear classifier on the concatenated embeddings. This reduces the effective parameter count from 16*256*65 ≈ 266k to 16*256*E + 16*E*65 (with E=32: ~164k), acting as a structured regularizer that captures byte similarity. It should exceed label-linear's 0.3375 dev accuracy while keeping inference cost low (a few thousand FLOPs/item). Failure: dev accuracy ≤ 0.3375 or Ir/item > 20000.
```

```python
"""Factorized linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Learns per-position byte embeddings (dim E) and a linear output layer.
Exported weights: magic 0x46414354, d, E, H, C; float32 embeddings[d][256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension
H = 0           # hidden size (0 = linear)
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x46414354  # 'FACT'

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    embeddings = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # W: (d*E, C)  Xavier uniform
    limit = math.sqrt(6.0 / (d * E + C))
    W = (rng.uniform(-limit, limit, (d * E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [embeddings, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: concatenate embeddings
            Z = np.empty((B, d * E), dtype=np.float32)
            for j in range(d):
                Z[:, j*E:(j+1)*E] = embeddings[j, Xb[:, j]]

            logits = Z @ W + b                     # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Loss gradient
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # Backward
            dW = Z.T @ dlogits                     # (d*E, C)
            db = dlogits.sum(axis=0)               # (C,)
            dZ = dlogits @ W.T                     # (B, d*E)

            # Gradient for embeddings
            dembeddings = np.zeros_like(embeddings)
            for j in range(d):
                dZ_j = dZ[:, j*E:(j+1)*E]
                np.add.at(dembeddings[j], Xb[:, j], dZ_j)

            # L2 regularization
            dW += L2 * W
            db += L2 * b
            dembeddings += L2 * embeddings

            grads = [dembeddings, dW, db]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Header: magic, d, E, H, C (5 int32)
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = embeddings.astype("<f4").tobytes()
    # W: (d*E) * C floats
    W_bytes = W.astype("<f4").tobytes()
    # b: C floats
    b_bytes = b.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(emb_bytes)
        f.write(W_bytes)
        f.write(b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x46414354, d, E, H, C; float32 embeddings[d][256][E], W[d*E][C], b[C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 20 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (H != 0) { fprintf(stderr, "hidden layer not supported\n"); return 3; }
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(acc + j * E, src, E * sizeof(float));
        }

        // Compute scores = acc @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * E; idx++) {
            float a = acc[idx];
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += a * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```