```hypothesis
Family: label-hash2 (linear model on hashed unigrams and bigrams).
Why: Tinyshakespeare next-char prediction benefits from modeling local character interactions (bigrams). A linear model on hashed unigram+bigram features captures these pairwise dependencies while remaining fast at inference (O(d^2) hash computations per item). It should beat label-linear (0.3375) by adding bigram features, and be faster than label-gbdt (131k Ir/item) due to simple linear scoring.
Failure signal: If dev accuracy <= 0.34, the hash collisions or linear limitation prevent effective bigram modeling; a larger hash or non-linear model would be needed.
```

```python
"""label-hash2: linear model on hashed unigrams and bigrams for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hash size 2^17 (131072). Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic 0x48415348, hash_bits, d, C; float32 W[hash_size][C], b[C]."""
import json, math, struct, sys
from collections import defaultdict

import numpy as np

HASH_BITS = 17
HASH_SIZE = 1 << HASH_BITS
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def mix32(x):
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = x ^ (x >> 16)
    return x & 0xFFFFFFFF

def hash_unigram(pos, byte):
    key = (pos << 8) | byte
    return mix32(key) & (HASH_SIZE - 1)

def hash_bigram(pos1, pos2, byte1, byte2):
    key = ((pos1 << 4) | pos2) << 16 | (byte1 << 8) | byte2
    return mix32(key) & (HASH_SIZE - 1)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Precompute feature indices for each example (16 unigrams + 120 bigrams = 136)
    feat_indices = []
    for i in range(n):
        xi = X[i]
        feats = []
        for p in range(d):
            feats.append(hash_unigram(p, xi[p]))
        for p1 in range(d):
            for p2 in range(p1 + 1, d):
                feats.append(hash_bigram(p1, p2, xi[p1], xi[p2]))
        feat_indices.append(np.array(feats, dtype=np.int32))

    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, size=(HASH_SIZE, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            bs = len(idx_batch)
            if bs == 0:
                continue

            # Sparse gradient accumulation via dict
            grad_W_dict = {}
            grad_b = np.zeros(C, dtype=np.float32)

            for idx in idx_batch:
                feats = feat_indices[idx]
                logits = W[feats].sum(axis=0) + b
                # softmax
                logits -= logits.max()
                exp_logits = np.exp(logits, dtype=np.float32)
                prob = exp_logits / exp_logits.sum()
                grad = prob
                grad[y[idx]] -= 1.0
                grad /= bs

                for f in feats:
                    if f in grad_W_dict:
                        grad_W_dict[f] += grad
                    else:
                        grad_W_dict[f] = grad.copy()
                grad_b += grad

            # Adam update on used rows only
            t += 1
            for f, g in grad_W_dict.items():
                m = m_W[f]
                v = v_W[f]
                m *= BETA1; m += (1 - BETA1) * g
                v *= BETA2; v += (1 - BETA2) * g * g
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                W[f] -= LR * m_hat / (np.sqrt(v_hat) + EPS)
                # weight decay (decoupled)
                W[f] *= (1 - LR * L2)
                m_W[f] = m
                v_W[f] = v

            # bias update
            m_b *= BETA1; m_b += (1 - BETA1) * grad_b
            v_b *= BETA2; v_b += (1 - BETA2) * grad_b * grad_b
            m_hat_b = m_b / (1 - BETA1 ** t)
            v_hat_b = v_b / (1 - BETA2 ** t)
            b -= LR * m_hat_b / (np.sqrt(v_hat_b) + EPS)

    # Write weights
    out = struct.pack("<4I", 0x48415348, HASH_BITS, d, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-hash2 infer: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x48415348), hash_bits, d, C; float W[hash_size][C], b[C]. */
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

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return x;
}

static inline uint32_t hash_unigram(int pos, int byte, uint32_t mask) {
    uint32_t key = (pos << 8) | byte;
    return mix32(key) & mask;
}

static inline uint32_t hash_bigram(int pos1, int pos2, int byte1, int byte2, uint32_t mask) {
    uint32_t key = ((pos1 << 4) | pos2) << 16 | (byte1 << 8) | byte2;
    return mix32(key) & mask;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    int hash_bits = hd[1];
    int d = hd[2];
    int C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    uint32_t hash_size = 1u << hash_bits;
    uint32_t mask = hash_size - 1;
    size_t expected = 16 + (size_t)hash_size * C * 4 + C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (size_t)hash_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b[c];

        // unigrams
        for (int p = 0; p < d; p++) {
            uint32_t h = hash_unigram(p, xi[p], mask);
            const float *w = W + (size_t)h * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }
        // bigrams
        for (int p1 = 0; p1 < d; p1++) {
            for (int p2 = p1 + 1; p2 < d; p2++) {
                uint32_t h = hash_bigram(p1, p2, xi[p1], xi[p2], mask);
                const float *w = W + (size_t)h * C;
                for (int c = 0; c < C; c++) acc[c] += w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```