```hypothesis
Family: label-hashbigram (linear model with unigram and hashed adjacent bigram features).
Why: tinyshakespeare character prediction benefits from local n-gram statistics; label-linear uses only unigrams (positional), missing pairwise interactions. Adding 15 adjacent bigrams hashed to 16k buckets captures local context with modest parameter increase (1.3M vs 0.27M) and fast inference (31 feature lookups per item).
Failure signal: dev accuracy <= 0.3375 (label-linear) or inference cost > 2x label-linear.
```
```python
"""label-hashbigram: linear model with unigram and hashed adjacent bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Features: 16 unigrams (pos*256 + byte) + 15 bigrams (hash(pos, byte1, byte2) mod 16384).
Total features: 4096 + 16384 = 20480. Output: 65 classes.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x48424C31, d, C, n_unigram, n_bigram_buckets; float32 bias[C]; float32 W[20480][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
HASH_BITS = 14
HASH_SIZE = 1 << HASH_BITS
HASH_MASK = HASH_SIZE - 1
MAGIC = 0x48424C31  # "HBL1"

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_bigram(pos, b1, b2):
    # 32-bit mix matching C implementation exactly
    h = (pos * 257 + b1) * 257 + b2
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return h & HASH_MASK

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16
    assert C == 65

    X_bytes = b"".join(records(sys.argv[2]))
    n = len(X_bytes) // d
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    assert len(y) == n

    # Precompute feature indices for all examples
    n_unigram = d * 256  # 4096
    n_bigram = HASH_SIZE  # 16384
    n_feat = n_unigram + n_bigram  # 20480
    feat_idx = np.zeros((n, d + d - 1), dtype=np.int32)  # 16 unigram + 15 bigram
    for i in range(n):
        xi = X[i]
        # unigrams
        for j in range(d):
            feat_idx[i, j] = j * 256 + xi[j]
        # bigrams (adjacent pairs)
        for j in range(d - 1):
            h = hash_bigram(j, xi[j], xi[j + 1])
            feat_idx[i, d + j] = n_unigram + h

    # Initialize weights
    rng = np.random.default_rng(SEED)
    scale = math.sqrt(6.0 / n_feat)
    W = rng.uniform(-scale, scale, (n_feat, C)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            B = len(idx)
            # Gather logits: (B, 31, C) -> sum over 31 -> (B, C)
            logits = W[feat_idx[idx]].sum(axis=1) + bias
            # Softmax
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient
            grad_logits = probs
            grad_logits[np.arange(B), y[idx]] -= 1.0
            grad_logits /= B
            # Grad W: scatter-add
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, feat_idx[idx].ravel(), np.repeat(grad_logits, feat_idx.shape[1], axis=0))
            grad_W += L2 * W
            # Grad bias
            grad_b = grad_logits.sum(axis=0)
            # Adam update
            t += 1
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # bias
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Write weights
    out = struct.pack("<5i", MAGIC, d, C, n_unigram, n_bigram)
    out += bias.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-hashbigram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x48424C31, d, C, n_unigram, n_bigram; float32 bias[C]; float32 W[n_unigram+n_bigram][C]. */
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

static inline uint32_t hash_bigram(int pos, uint8_t b1, uint8_t b2) {
    uint32_t h = (pos * 257 + b1) * 257 + b2;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & 16383; // HASH_MASK = (1<<14)-1
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x48424C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_unigram = hd[3], n_bigram = hd[4];
    if (d != 16 || C != 65 || n_unigram != 4096 || n_bigram != 16384) {
        fprintf(stderr, "bad weights header\n"); return 3; }
    int n_feat = n_unigram + n_bigram; // 20480
    long expected = 20 + (long)C * 4 + (long)n_feat * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 5);
    const float *W = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize with bias
        for (int c = 0; c < C; c++) acc[c] = bias[c];

        // Unigrams
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }

        // Bigrams (adjacent pairs)
        for (int j = 0; j < d - 1; j++) {
            uint32_t h = hash_bigram(j, xi[j], xi[j + 1]);
            int idx = n_unigram + h;
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }

        // Argmax
        best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```