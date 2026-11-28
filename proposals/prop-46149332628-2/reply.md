```hypothesis
Family: label-linear-ngram (linear model with hashed bigram features).
Why: Character-level language modeling benefits from local n-gram statistics; adding position-aware bigram features via feature hashing captures pairwise interactions with minimal parameter overhead (4096 extra weights). The linear structure keeps inference fast (O(d + d) lookups).
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear.
```
```python
"""Linear model with hashed bigram features for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Format: magic=0x4E47524C, d, H=0, C, hash_size; T_uni[d][256][C]; T_bi[hash_size][C]; b[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
HASH_SIZE = 4096          # power of two
LOG2_HASH = 12            # log2(HASH_SIZE)
MAGIC = 0x4E47524C        # 'NGRL'
KNUTH64 = 11400714819323198485


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def bigram_hash_vec(pos, b1, b2):
    """Vectorized 64-bit multiplicative hash, returns uint64 in [0, HASH_SIZE)."""
    h = (np.uint64(pos) * 65536 + b1.astype(np.uint64) * 256 + b2.astype(np.uint64))
    h = (h * np.uint64(KNUTH64)) >> (64 - LOG2_HASH)
    return h.astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: W_uni (d*256, C), W_bi (HASH_SIZE, C), b (C,)
    W_uni = rng.uniform(-1, 1, (d * 256, C)).astype(np.float32) * math.sqrt(6.0 / (d * 256))
    W_bi = rng.uniform(-1, 1, (HASH_SIZE, C)).astype(np.float32) * math.sqrt(6.0 / HASH_SIZE)
    b = np.zeros(C, np.float32)
    params = [W_uni, W_bi, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            scores = np.tile(b, (B, 1))
            # Unigrams
            for j in range(d):
                scores += W_uni[j * 256 + xb[:, j]]
            # Bigrams
            for j in range(d - 1):
                h = bigram_hash_vec(j, xb[:, j], xb[:, j + 1])
                scores += W_bi[h]

            # Softmax + gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            dW_uni = np.zeros_like(W_uni)
            dW_bi = np.zeros_like(W_bi)
            db = grad.sum(axis=0)

            for j in range(d):
                np.add.at(dW_uni, j * 256 + xb[:, j], grad)
            for j in range(d - 1):
                h = bigram_hash_vec(j, xb[:, j], xb[:, j + 1])
                np.add.at(dW_bi, h, grad)

            # L2
            dW_uni += L2 * W_uni
            dW_bi += L2 * W_bi

            grads = [dW_uni, dW_bi, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fold unigram weights into per-position tables T_uni[d][256][C]
    T_uni = W_uni.reshape(d, 256, C).astype("<f4")
    T_bi = W_bi.astype("<f4")
    b = b.astype("<f4")

    out = struct.pack("<5i", MAGIC, d, 0, C, HASH_SIZE)
    out += T_uni.tobytes() + T_bi.tobytes() + b.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Linear model with hashed bigram features: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: 1 byte class index.
 * Weights format: magic(0x4E47524C), d, H=0, C, hash_size; T_uni[d][256][C]; T_bi[hash_size][C]; b[C] (float32 LE). */
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

static inline uint32_t bigram_hash(int pos, uint8_t b1, uint8_t b2, int log2_hash) {
    uint64_t h = (uint64_t)pos * 65536 + (uint64_t)b1 * 256 + b2;
    h *= 11400714819323198485ULL;
    return (uint32_t)(h >> (64 - log2_hash));
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4E47524C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], hash_size = hd[4];
    if (H != 0 || C < 1 || C > 256 || (hash_size & (hash_size - 1)) != 0) {
        fprintf(stderr, "bad header\n"); return 3;
    }
    int log2_hash = 0; while ((1 << log2_hash) < hash_size) log2_hash++;
    long expected = 20L + 4L * ((long)d * 256 * C + (long)hash_size * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_uni = (const float *)(hd + 5);
    const float *T_bi = T_uni + (long)d * 256 * C;
    const float *b = T_bi + (long)hash_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *t = T_uni + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        for (int j = 0; j < d - 1; j++) {
            uint32_t h = bigram_hash(j, xi[j], xi[j + 1], log2_hash);
            const float *t = T_bi + (long)h * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```