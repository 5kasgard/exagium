```hypothesis
Family: label-hash (feature-hashed linear model with unigrams and bigrams).
Why: Character-level language modeling benefits from bigram interactions; hashing compresses the large bigram space (15*256^2) into a fixed table (D=8192) allowing a linear model to capture pairwise context without exploding parameters. Adam optimization on the hashed features should generalize better than pure unigram linear (label-linear) while keeping inference fast (31 table lookups per item vs 16).
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 15000 (2× label-linear).
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 8192          # hash table size (power of two)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_id(id_val, mask):
    # 64-bit multiplicative hash, same as in infer.c
    return (id_val * 11400714819323198485) & mask

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    assert n == y.shape[0]

    # Precompute hashed feature indices for all records
    mask = D - 1
    indices = np.zeros((n, 31), dtype=np.int32)  # 16 unigrams + 15 bigrams
    for i in range(n):
        rec = X[i]
        idx = 0
        # unigrams
        for p in range(d):
            indices[i, idx] = hash_id(p * 256 + rec[p], mask)
            idx += 1
        # bigrams
        for p in range(d - 1):
            indices[i, idx] = hash_id(d * 256 + p * 256 * 256 + rec[p] * 256 + rec[p + 1], mask)
            idx += 1

    # Model parameters
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            batch_size = len(batch_idx)
            batch_indices = indices[batch_idx]   # (batch_size, 31)
            batch_y = y[batch_idx]

            # Forward: scores = bias + sum_{features} W[feature]
            scores = np.tile(bias, (batch_size, 1))
            for f in range(31):
                scores += W[batch_indices[:, f]]

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), batch_y] -= 1.0
            probs /= batch_size

            # Bias gradient
            grad_b = probs.sum(axis=0)

            # Weight gradient (sparse)
            flat_idx = batch_indices.ravel()
            flat_grad = np.repeat(probs, 31, axis=0)
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, flat_idx, flat_grad)

            # L2 regularization only on touched rows (approximation)
            unique_idxs = np.unique(flat_idx)
            grad_W[unique_idxs] += L2 * W[unique_idxs]

            # Adam update for weights (only touched rows)
            t += 1
            m_W[unique_idxs] = beta1 * m_W[unique_idxs] + (1 - beta1) * grad_W[unique_idxs]
            v_W[unique_idxs] = beta2 * v_W[unique_idxs] + (1 - beta2) * (grad_W[unique_idxs] ** 2)
            m_hat = m_W[unique_idxs] / (1 - beta1 ** t)
            v_hat = v_W[unique_idxs] / (1 - beta2 ** t)
            W[unique_idxs] -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # Adam update for bias (dense)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Write weights: magic, d, D, C; then W (D*C float32), bias (C float32)
    magic = 0x48415348  # "HASH"
    header = struct.pack("<4i", magic, d, D, C)
    out = header + W.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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

static inline uint32_t hash_id(uint64_t id, uint32_t mask) {
    return (id * 11400714819323198485ULL) & mask;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint32_t D_mask = D - 1;
    float *W = (float *)(hd + 4);
    float *bias = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        // unigrams
        for (int p = 0; p < d; p++) {
            uint64_t id = (uint64_t)p * 256 + xi[p];
            uint32_t idx = hash_id(id, D_mask);
            float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        // bigrams
        for (int p = 0; p < d - 1; p++) {
            uint64_t id = (uint64_t)d * 256 + p * 256 * 256 + xi[p] * 256 + xi[p + 1];
            uint32_t idx = hash_id(id, D_mask);
            float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```