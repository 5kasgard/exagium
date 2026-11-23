```hypothesis
Family: label-hashngram. A linear model on hashed unigram and bigram features (31 features per 16-byte context) with a fixed hash table of 4096 buckets. Captures local character interactions crucial for language modeling while keeping inference fast (single pass over features, contiguous memory access). Should beat label-linear (0.3375) because bigrams add predictive power without increasing parameter count. Failure would show if hash collisions degrade performance or if linear capacity is insufficient.
```

```python
"""Hash n-gram family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Linear model on hashed unigrams and bigrams.
Hash table size 4096, 65 classes. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))."""
import json, math, struct, sys

import numpy as np

HASH_SIZE = 4096
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x484E4731  # 'HNG1' little-endian
KNUTH = 2654435761  # 32-bit multiplicative constant
MASK = HASH_SIZE - 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_unigram(pos, byte):
    key = (pos << 8) | byte
    return (key * KNUTH) & MASK

def hash_bigram(pos, b1, b2):
    key = 0x10000 | (pos << 16) | (b1 << 8) | b2
    return (key * KNUTH) & MASK

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (HASH_SIZE, C)) * math.sqrt(6.0 / HASH_SIZE)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)

            # compute 31 feature hashes per example (vectorized)
            indices = np.empty((B, 31), dtype=np.int32)
            for f in range(31):
                if f < 16:
                    pos = f
                    key = (pos << 8) | Xb[:, pos]
                else:
                    pos = f - 16
                    key = 0x10000 | (pos << 16) | (Xb[:, pos] << 8) | Xb[:, pos + 1]
                indices[:, f] = (key * KNUTH) & MASK

            # forward: scores = sum_{features} W[hash] + b
            scores = W[indices].sum(axis=1) + b  # (B, C)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = (probs / B).astype(np.float32)  # (B, C)

            # gradients
            grad_W = np.zeros_like(W)
            grad_b = grad_scores.sum(axis=0)
            for f in range(31):
                np.add.at(grad_W, indices[:, f], grad_scores)
            grad_W += L2 * W
            grads = [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, HASH_SIZE, C, d)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram linear model for label tasks: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (argmax). */
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

static inline uint32_t hash_key(uint32_t key) {
    return (key * 2654435761u) & 0xFFFu;  // HASH_SIZE=4096, mask=4095
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int HASH_SIZE = hd[1], C = hd[2], d = hd[3];
    if (C < 1 || C > 256 || d != 16 || nw != 16 + 4L * ((long)HASH_SIZE * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)HASH_SIZE * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        for (int p = 0; p < 16; p++) {
            uint32_t h = hash_key((p << 8) | xi[p]);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int p = 0; p < 15; p++) {
            uint32_t h = hash_key(0x10000 | (p << 16) | (xi[p] << 8) | xi[p + 1]);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```