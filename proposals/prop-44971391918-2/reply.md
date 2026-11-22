```hypothesis
Family: label-hashngram. A linear model on hashed n-gram features (unigrams, bigrams, trigrams) from the 16-byte context.
Why: Character-level language modeling benefits from local n-gram statistics; hashing compresses the large n-gram space into a fixed-size feature vector, allowing a fast linear model to capture pairwise/triple interactions that label-linear misses.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or CPU cost exceeds label-linear significantly, the hashing collisions or insufficient capacity hurt more than the n-gram information helps.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (would be chosen on the tuning split: last 20% of train.in)
F = 4096            # hash feature size
NGRAM_LENGTHS = [1, 2, 3]
BATCH = 64
LR = 0.01
L2 = 1e-4
EPOCHS = 50
SEED = 1

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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Deterministic seeds for each (length, position)
    rng = np.random.default_rng(SEED)
    seeds = []
    ngram_info = []  # list of (L, pos)
    for L in NGRAM_LENGTHS:
        for pos in range(d - L + 1):
            seeds.append(rng.integers(0, 2**32, dtype=np.uint32))
            ngram_info.append((L, pos))
    M = len(seeds)
    seeds = np.array(seeds, dtype=np.uint32)

    # Precompute feature indices for all training examples
    feat_indices = np.zeros((n, M), dtype=np.int32)
    for i in range(n):
        x = X[i]
        for m, (L, pos) in enumerate(ngram_info):
            h = seeds[m]
            for k in range(L):
                h = (h * 31 + x[pos + k]) & 0xFFFFFFFF
            feat_indices[i, m] = h % F

    # Initialize weights
    rng = np.random.default_rng(SEED + 1)
    W = (rng.normal(0, 0.01, size=(F, C))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    indices = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(indices)
        for start in range(0, n, BATCH):
            batch_idx = indices[start:start + BATCH]
            B = len(batch_idx)
            idx_batch = feat_indices[batch_idx]          # (B, M)
            # Scores: bias + sum_m W[idx_batch[:, m], :]
            scores = bias + np.sum(W[idx_batch], axis=1)  # (B, C)
            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), y[batch_idx]] -= 1.0
            grad = probs / B                              # (B, C)

            # Accumulate gradient for W
            g_W = np.zeros_like(W)
            for m in range(M):
                np.add.at(g_W, idx_batch[:, m], grad)
            g_W += L2 * W

            # Bias gradient
            g_b = grad.sum(axis=0)

            # Adam step
            t += 1
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_b = beta1 * m_b + (1 - beta1) * g_b
            v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # Write weights.bin
    # Magic: 0x484E4731 ('HNG1'), then F, C, M, d (all int32)
    # Followed by M uint32 seeds, C float32 bias, F*C float32 W (row-major)
    out = struct.pack("<5i", 0x484E4731, F, C, M, d)
    out += seeds.astype("<u4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x484E4731) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t F = hd[1], C = hd[2], M = hd[3], d = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    uint32_t *seeds = hd + 5;
    float *bias = (float *)(seeds + M);
    float *W = bias + C;
    long expected = 20 + 4L * M + 4L * C + 4L * F * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        memcpy(scores, bias, C * sizeof(float));
        for (uint32_t m = 0; m < M; m++) {
            uint32_t h = seeds[m];
            // ngram_info order: lengths 1,2,3 each with positions 0..d-L
            // We recompute the same order as in train.py to know which bytes to hash.
            // Instead of storing L and pos, we can recompute them from m.
            // But we stored only seeds. We need to know L and pos for each m.
            // We'll embed the ngram_info in the weights or recompute deterministically.
            // Since NGRAM_LENGTHS and d are fixed, we can recompute.
            // We'll hardcode the same logic here.
        }
        // We'll restructure: store L and pos for each m in the weights file to avoid duplication.
        // But the contract says we can choose the format. Let's adjust train.py to also write L and pos arrays.
        // However, to keep the change minimal, we can recompute the mapping from m to (L,pos) in C using the same constants.
        // We'll do that.
    }
    // The above is incomplete; we need to finish the inference loop with the hash computation.
    // Let's rewrite the infer.c completely with the correct logic.
    return 0;
}
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x484E4731) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t F = hd[1], C = hd[2], M = hd[3], d = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    uint32_t *seeds = hd + 5;
    float *bias = (float *)(seeds + M);
    float *W = bias + C;
    long expected = 20 + 4L * M + 4L * C + 4L * F * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    // Constants matching train.py
    const int NGRAM_LENGTHS[] = {1, 2, 3};
    const int NUM_LENGTHS = 3;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        memcpy(scores, bias, C * sizeof(float));
        int m = 0;
        for (int li = 0; li < NUM_LENGTHS; li++) {
            int L = NGRAM_LENGTHS[li];
            for (int p = 0; p <= d - L; p++) {
                uint32_t h = seeds[m];
                for (int o = 0; o < L; o++) {
                    h = h * 31 + xi[p + o];
                }
                uint32_t idx = h % F;
                const float *wrow = W + (size_t)idx * C;
                for (uint32_t c = 0; c < C; c++) scores[c] += wrow[c];
                m++;
            }
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```