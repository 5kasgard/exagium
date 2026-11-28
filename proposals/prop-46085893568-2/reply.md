```hypothesis
The label-hashngram family uses a linear model on hashed n-gram features (unigrams, bigrams, trigrams) extracted from the 16-byte context. This captures local character interactions (e.g., common bigrams/trigrams in English) that a pure linear model on single bytes misses, while keeping inference extremely fast (only ~45 hash lookups and vector additions per item). The model is trained with Adam on the full training set after selecting bucket count and learning rate on the last 20% tuning split. If dev accuracy does not exceed 0.34, the hashed n-gram approach is insufficient for this task.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
B_CANDIDATES = [10000, 20000, 40000]
LR_CANDIDATES = [0.003, 0.01, 0.03]
NGRAM_MAX = 3
SEED = 1
L2 = 1e-5
BATCH = 256
EPOCHS = 15  # enough for convergence on small data

MAGIC = 0x484E4731  # "HNG1"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_ngram(ngram_bytes, length, B):
    # DJB2 hash (32-bit)
    h = 5381 + length
    for b in ngram_bytes:
        h = (h * 33 + b) & 0xFFFFFFFF
    return h % B


def extract_buckets(x, B, ngram_max):
    """Return list of bucket indices for all n-grams in x (16 bytes)."""
    buckets = []
    for n in range(1, ngram_max + 1):
        for i in range(0, 17 - n):
            buckets.append(hash_ngram(x[i:i + n], n, B))
    return buckets


def train_model(X, y, B, C, lr, rng, epochs=EPOCHS):
    """Train linear model on hashed n-gram features using Adam."""
    W = np.zeros((B, C), dtype=np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            grad = np.zeros_like(W)
            # Forward + backward per example (batch small enough for Python loop)
            for i in range(len(xb)):
                buckets = extract_buckets(xb[i], B, NGRAM_MAX)
                scores = W[buckets].sum(axis=0)
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                probs[yb[i]] -= 1.0
                for b in buckets:
                    grad[b] += probs
            grad /= len(xb)
            grad += L2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W


def evaluate(X, y, W, B):
    correct = 0
    for i in range(len(X)):
        buckets = extract_buckets(X[i], B, NGRAM_MAX)
        scores = W[buckets].sum(axis=0)
        if scores.argmax() == y[i]:
            correct += 1
    return correct / len(X)


def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.array([list(r) for r in X_recs], dtype=np.uint8)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    C = card["classes"]
    N = X.shape[0]

    # Split: last 20% for tuning
    split = int(N * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Grid search on tuning split
    best_acc = -1.0
    best_B, best_lr = B_CANDIDATES[0], LR_CANDIDATES[0]
    rng = np.random.default_rng(SEED)
    for B in B_CANDIDATES:
        for lr in LR_CANDIDATES:
            W = train_model(X_train, y_train, B, C, lr, rng, epochs=EPOCHS)
            acc = evaluate(X_tune, y_tune, W, B)
            if acc > best_acc:
                best_acc = acc
                best_B, best_lr = B, lr

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    W_final = train_model(X, y, best_B, C, best_lr, rng, epochs=EPOCHS)

    # Write weights: magic, B, C, ngram_max, then W (B x C row-major float32)
    out = struct.pack("<4i", MAGIC, best_B, C, NGRAM_MAX)
    out += W_final.astype("<f4").tobytes()
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

/* DJB2 hash (32-bit) */
static uint32_t hash_ngram(const unsigned char *bytes, int len, uint32_t B) {
    uint32_t h = 5381 + (uint32_t)len;
    for (int i = 0; i < len; i++) {
        h = h * 33 + bytes[i];
    }
    return h % B;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int B = hd[1], C = hd[2], ngram_max = hd[3];
    if (C < 1 || C > 256 || ngram_max < 1 || ngram_max > 3 ||
        nw != 16 + 4L * (long)B * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        memset(scores, 0, C * sizeof(float));
        for (int n = 1; n <= ngram_max; n++) {
            for (int start = 0; start <= 16 - n; start++) {
                uint32_t bucket = hash_ngram(xi + start, n, (uint32_t)B);
                const float *w = W + (long)bucket * C;
                for (int c = 0; c < C; c++) scores[c] += w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```