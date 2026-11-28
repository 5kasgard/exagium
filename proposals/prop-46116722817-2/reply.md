```hypothesis
We propose a "hashed n-gram embedding" model (hash-embed-linear). It maps all n-grams of length 1–4 from the 16-byte context into a fixed-size embedding table (64K buckets, 32-dim) via FNV-1a hashing, sums the embeddings, and applies a linear layer to predict the next character. This captures local sequential dependencies better than unigram linear models while staying fast and compact. If dev accuracy does not exceed label-linear (0.3375) or inference cost per item is significantly higher, the approach fails.
```

```python
"""Hash-embed-linear: hashed n-gram embeddings + linear layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, not tuned on the tuning split)
D = 65536          # hash table size (power of two)
E = 32             # embedding dimension
ORDERS = (1, 2, 3, 4)   # n-gram lengths
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
SEED = 1

FNV_OFFSET = 14695981039346656037
FNV_PRIME  = 1099511628211
MASK64 = (1 << 64) - 1
MASK_BUCKET = D - 1

def fnv_hash_bytes(buf, length, pos):
    h = FNV_OFFSET
    for i in range(length):
        h ^= buf[i]
        h = (h * FNV_PRIME) & MASK64
    h ^= length
    h = (h * FNV_PRIME) & MASK64
    h ^= pos
    h = (h * FNV_PRIME) & MASK64
    return h & MASK_BUCKET

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
    N = X.shape[0]

    # Precompute bucket lists for every example
    buckets_list = []
    for i in range(N):
        ctx = X[i]
        buckets = []
        for L in ORDERS:
            for pos in range(0, d - L + 1):
                h = fnv_hash_bytes(ctx[pos:pos+L], L, pos)
                buckets.append(h)
        buckets_list.append(np.array(buckets, dtype=np.uint32))

    # Initialize parameters
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0, 1/math.sqrt(E), size=(D, E)).astype(np.float32)
    W   = rng.normal(0, 1/math.sqrt(E), size=(E, C)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W,   v_W   = np.zeros_like(W),   np.zeros_like(W)
    m_b,   v_b   = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            bs = len(idx)
            # Sum embeddings for each example
            sum_emb = np.zeros((bs, E), dtype=np.float32)
            for b, i in enumerate(idx):
                sum_emb[b] = emb[buckets_list[i]].sum(axis=0)

            # Forward
            scores = sum_emb @ W + bias
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Gradient
            grad_scores = probs
            grad_scores[np.arange(bs), y[idx]] -= 1.0
            grad_scores /= bs

            grad_W = sum_emb.T @ grad_scores
            grad_bias = grad_scores.sum(axis=0)
            grad_sum_emb = grad_scores @ W.T

            grad_emb = np.zeros_like(emb)
            for b, i in enumerate(idx):
                np.add.at(grad_emb, buckets_list[i], grad_sum_emb[b])

            # L2
            grad_emb += L2 * emb
            grad_W   += L2 * W

            # Adam update
            t += 1
            for param, grad, m, v in (
                (emb, grad_emb, m_emb, v_emb),
                (W,   grad_W,   m_W,   v_W),
                (bias, grad_bias, m_b,   v_b)
            ):
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Magic 0x48454C31 ("HEL1" little-endian), then D, E, C
    header = struct.pack("<4i", 0x48454C31, D, E, C)
    out = header + emb.astype("<f4").tobytes() + W.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Hash-embed-linear inference: ./infer weights.bin in.rec N
 * Each input record = 16 bytes context. Output = 1 byte class index. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

/* FNV-1a 64-bit, same as train.py */
static uint32_t fnv_hash(const unsigned char *buf, int len, int pos, uint32_t mask) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)buf[i];
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)len;
    h *= 1099511628211ULL;
    h ^= (uint64_t)pos;
    h *= 1099511628211ULL;
    return (uint32_t)(h & mask);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48454C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int D = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }

    const int orders[] = {1,2,3,4};
    const int n_orders = 4;
    const int d = 16;

    size_t expected = 16 + 4ull*( (size_t)D*E + (size_t)E*C + C );
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W   = emb + (size_t)D * E;
    const float *bias = W   + (size_t)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(E, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(sum_emb, 0, E * sizeof(float));
        for (int o = 0; o < n_orders; o++) {
            int L = orders[o];
            for (int p = 0; p <= d - L; p++) {
                uint32_t bucket = fnv_hash(xi + p, L, p, D-1);
                const float *e = emb + (size_t)bucket * E;
                for (int j = 0; j < E; j++) sum_emb[j] += e[j];
            }
        }

        for (int c = 0; c < C; c++) {
            float acc = bias[c];
            for (int j = 0; j < E; j++) acc += sum_emb[j] * W[(size_t)j * C + c];
            scores[c] = acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```