```hypothesis
We propose a hashed n-gram linear model (FastText-style) that sums embeddings of all character n-grams (lengths 1–4) in the 16-byte context. This captures local sequential statistics crucial for language modeling while remaining a single linear layer over a fixed hash table, so inference is fast (58 hash lookups + 58*65 FMA per item). It should beat label-linear (0.3375) because n-grams model character interactions that independent per-position one-hots cannot. If it fails, it will show that hashing collisions destroy signal or that n-gram features alone are insufficient for this task.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Features: all contiguous n-grams of length 1..4 from the 16-byte context, hashed to 65536 buckets.
Model: bias[C] + sum_{ngram} W[hash(ngram)][C]. Trained with Adam (lr=0.03, L2=1e-4, batch=64).
Weights format: uint32 magic(0x4E47524D), num_buckets, C; float32 bias[C]; float32 W[num_buckets*C] (row-major)."""
import json, math, struct, sys

import numpy as np

NUM_BUCKETS = 65536
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.03, 1e-4, 64, 600000, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_ngram(ngram):
    h = 0
    for b in ngram:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h & (NUM_BUCKETS - 1)


def get_buckets(ctx):
    """Return list of 58 bucket indices for all n-grams length 1..4 in 16-byte ctx."""
    buckets = []
    for L in range(1, 5):
        for start in range(0, 17 - L):
            buckets.append(hash_ngram(ctx[start:start + L]))
    return buckets


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Precompute buckets for all examples
    all_buckets = [get_buckets(X[i]) for i in range(n)]

    W = rng.normal(0, 0.01, (NUM_BUCKETS, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)
            scores = np.tile(b, (bs, 1))
            for i, ex in enumerate(idx):
                scores[i] += W[all_buckets[ex]].sum(axis=0)

            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(bs), y[idx]] -= 1.0
            grad_scores /= bs

            grad_b = grad_scores.sum(axis=0)
            grad_W_dict = {}
            for i, ex in enumerate(idx):
                g = grad_scores[i]
                for bucket in all_buckets[ex]:
                    if bucket in grad_W_dict:
                        grad_W_dict[bucket] += g
                    else:
                        grad_W_dict[bucket] = g.copy()

            t += 1
            # bias update
            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            m_b_hat = m_b / (1 - 0.9 ** t)
            v_b_hat = v_b / (1 - 0.999 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

            # weight updates
            for bucket, g in grad_W_dict.items():
                g_reg = g + L2 * W[bucket]
                m_W[bucket] = 0.9 * m_W[bucket] + 0.1 * g_reg
                v_W[bucket] = 0.999 * v_W[bucket] + 0.001 * g_reg * g_reg
                m_hat = m_W[bucket] / (1 - 0.9 ** t)
                v_hat = v_W[bucket] / (1 - 0.999 ** t)
                W[bucket] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x4E47524D  # 'NGRM'
    out = struct.pack("<III", magic, NUM_BUCKETS, C)
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights: uint32 magic(0x4E47524D), num_buckets, C; float bias[C]; float W[num_buckets*C]. */
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

static uint32_t hash_ngram(const unsigned char *ngram, int len) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) h = h * 31 + ngram[i];
    return h & 0xFFFF;  // NUM_BUCKETS = 65536
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t num_buckets = hd[1], C = hd[2];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 12 + 4L * C + 4L * num_buckets * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 3);
    const float *W = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // all n-grams length 1..4
        for (int L = 1; L <= 4; L++) {
            for (int start = 0; start <= 16 - L; start++) {
                uint32_t bucket = hash_ngram(xi + start, L);
                const float *w = W + (long)bucket * C;
                for (int c = 0; c < C; c++) scores[c] += w[c];
            }
        }

        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```