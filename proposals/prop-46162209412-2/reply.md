```hypothesis
Family: label-hashngram. A linear model over hashed unigram and bigram features (feature hashing). The 16-byte context has strong local pairwise dependencies (e.g., "th", "he") that a unigram linear model misses. Hashing bigrams into a fixed-size table (e.g., 2048) captures these interactions with a small, constant memory footprint. Inference requires 31 hash computations and 31*65 additions per item, comparable to label-linear cost. If dev accuracy does not exceed label-linear's 0.3375, the family fails.
```

```python
"""label-hashngram: linear model over hashed unigram+bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x484E4731, hash_size, C, d; float32 W[hash_size][C], b[C]."""
import json, math, struct, sys

import numpy as np

HASH_SIZE = 2048
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
HASH_SEED = 0x9e3779b97f4a7c15  # fixed hash seed


def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x


def hash_idx(key):
    return (splitmix64(key + HASH_SEED) % HASH_SIZE).astype(np.int32)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Precompute feature indices for all examples: 16 unigrams + 15 bigrams = 31 features
    # unigram key: (pos << 8) | byte
    # bigram key: (pos << 16) | (byte1 << 8) | byte2
    unigram_keys = (np.arange(d, dtype=np.uint64) << 8) | X_bytes.T  # (d, n)
    bigram_keys = ((np.arange(d - 1, dtype=np.uint64) << 16) |
                   (X_bytes[:, :-1].T << 8) |
                   X_bytes[:, 1:].T)  # (d-1, n)
    unigram_idx = hash_idx(unigram_keys).T   # (n, d)
    bigram_idx = hash_idx(bigram_keys).T     # (n, d-1)
    indices = np.concatenate([unigram_idx, bigram_idx], axis=1).astype(np.int32)  # (n, 31)

    rng = np.random.default_rng(SEED)
    # Initialize weights
    scale = math.sqrt(6.0 / HASH_SIZE)
    W = (rng.uniform(-1, 1, (HASH_SIZE, C)) * scale).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = indices[idx]          # (B, 31)
            yy = y[idx]                       # (B,)

            # Gather weights for active features: (B, 31, C)
            W_batch = W[batch_idx]
            logits = W_batch.sum(axis=1) + b  # (B, C)

            g = softmax_grad(logits, yy)      # (B, C)

            # Gradient for W: scatter-add g to each feature position
            dW = np.zeros_like(W)
            for f in range(31):
                np.add.at(dW, batch_idx[:, f], g)
            dW += L2 * W
            db = g.sum(axis=0) + L2 * b

            grads = [dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x484E4731  # "HNG1"
    out = struct.pack("<4i", magic, HASH_SIZE, C, d)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-hashngram infer: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E4731, hash_size, C, d; float32 W[hash_size][C], b[C]. */
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

/* splitmix64 hash */
static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static uint32_t hash_idx(uint64_t key, uint32_t hash_size, uint64_t hash_seed) {
    return (uint32_t)(splitmix64(key + hash_seed) % hash_size);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int hash_size = hd[1], C = hd[2], d = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)hash_size * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)hash_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    const uint64_t HASH_SEED = 0x9e3779b97f4a7c15ULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b[c];

        /* unigrams */
        for (int j = 0; j < d; j++) {
            uint64_t key = ((uint64_t)j << 8) | xi[j];
            uint32_t idx = hash_idx(key, hash_size, HASH_SEED);
            const float *row = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += row[c];
        }
        /* bigrams */
        for (int j = 0; j < d - 1; j++) {
            uint64_t key = ((uint64_t)j << 16) | ((uint64_t)xi[j] << 8) | xi[j + 1];
            uint32_t idx = hash_idx(key, hash_size, HASH_SEED);
            const float *row = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```