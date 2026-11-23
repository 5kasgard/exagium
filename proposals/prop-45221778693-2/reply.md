```hypothesis
We propose a hashed n-gram linear model (label-hashngram) that captures local sequential patterns
(unigrams, bigrams, trigrams) via a fixed-size hash table. This should outperform label-linear by
modeling character interactions, while remaining fast due to simple hash lookups and vector additions.
If it does not beat label-linear on dev accuracy, it would show that the task requires longer-range
dependencies or more complex modeling than local n-grams.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Features: unigrams, bigrams, trigrams at each position, hashed to 2^17 buckets.
Linear weights per bucket per class (65) + bias. Trained with Adam on 80/20 split,
hyperparameters chosen on validation split, then retrained on all data."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned on validation split)
SEED = 1
BUCKET_BITS = 17
BUCKETS = 1 << BUCKET_BITS          # 131072
BASE = 257                          # hash base (position + bytes)
NGRAM_ORDERS = (1, 2, 3)            # unigram, bigram, trigram
# Training config grid (will be evaluated on validation split)
GRID = [
    {"lr": 0.1, "epochs": 10},
    {"lr": 0.03, "epochs": 15},
    {"lr": 0.01, "epochs": 20},
    {"lr": 0.003, "epochs": 30},
]
BATCH = 64
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
L2 = 1e-5

MAGIC = 0x484E4731  # "HNG1"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def extract_features(x, d):
    """Return list of (hash_bucket) for a single 16-byte context."""
    feats = []
    for order in NGRAM_ORDERS:
        for pos in range(d - order + 1):
            # Compute hash: position * BASE^order + bytes
            h = pos
            for o in range(order):
                h = h * BASE + x[pos + o]
            feats.append(h & (BUCKETS - 1))  # modulo power of two
    return feats


def compute_scores(W, b, feats, C):
    """scores = b + sum_{f in feats} W[f]  (W shape: BUCKETS x C)"""
    s = b.copy()
    for f in feats:
        s += W[f]
    return s


def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    n_val = int(n * val_frac)
    n_train = n - n_val
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def accuracy(W, b, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        feats = extract_features(X[i], X.shape[1])
        scores = compute_scores(W, b, feats, y.max() + 1)
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n


def sgd_step(W, b, mW, vW, mb, vb, t, X_batch, y_batch, lr, l2):
    C = W.shape[1]
    n_batch = X_batch.shape[0]
    # gradients
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    for i in range(n_batch):
        feats = extract_features(X_batch[i], X_batch.shape[1])
        scores = compute_scores(W, b, feats, C)
        # softmax
        scores -= scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        probs[y_batch[i]] -= 1.0
        # accumulate gradients
        for f in feats:
            gW[f] += probs
        gb += probs
    gW /= n_batch
    gb /= n_batch
    gW += l2 * W  # L2 regularization on weights only
    # Adam update
    t += 1
    mW = BETA1 * mW + (1 - BETA1) * gW
    vW = BETA2 * vW + (1 - BETA2) * (gW * gW)
    mb = BETA1 * mb + (1 - BETA1) * gb
    vb = BETA2 * vb + (1 - BETA2) * (gb * gb)
    mW_hat = mW / (1 - BETA1 ** t)
    vW_hat = vW / (1 - BETA2 ** t)
    mb_hat = mb / (1 - BETA1 ** t)
    vb_hat = vb / (1 - BETA2 ** t)
    W -= lr * mW_hat / (np.sqrt(vW_hat) + EPS)
    b -= lr * mb_hat / (np.sqrt(vb_hat) + EPS)
    return W, b, mW, vW, mb, vb, t


def train_model(X, y, C, lr, epochs):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, (BUCKETS, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            W, b, mW, vW, mb, vb, t = sgd_step(W, b, mW, vW, mb, vb, t,
                                                 X[idx], y[idx], lr, L2)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]

    # Split for validation (last 20% in file order)
    (X_tr, y_tr), (X_val, y_val) = train_val_split(X, y, 0.2)

    # Grid search on validation split
    best_acc = -1.0
    best_cfg = None
    best_W, best_b = None, None
    for cfg in GRID:
        W, b = train_model(X_tr, y_tr, C, cfg["lr"], cfg["epochs"])
        acc = accuracy(W, b, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_W, best_b = W.copy(), b.copy()
        print(f"lr={cfg['lr']:.4f}, epochs={cfg['epochs']} -> val acc={acc:.4f}")

    print(f"Best config: {best_cfg}, val acc={best_acc:.4f}")

    # Retrain on full dataset with best config
    W_full, b_full = train_model(X, y, C, best_cfg["lr"], best_cfg["epochs"])

    # Write weights.bin
    # Format: magic (int32), BUCKETS (int32), C (int32), ngram_orders_bitmap (int32),
    # then W (BUCKETS x C float32, row-major), then b (C float32)
    ngram_bitmap = 0
    for o in NGRAM_ORDERS:
        ngram_bitmap |= 1 << o
    header = struct.pack("<4i", MAGIC, BUCKETS, C, ngram_bitmap)
    out = header + W_full.astype("<f4").tobytes() + b_full.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E4731, buckets, classes, ngram_bitmap,
 * float32 W[buckets][classes], float32 bias[classes]. */
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

/* Hash function matching Python: position * BASE^order + bytes, modulo BUCKETS (power of two). */
#define BASE 257
#define BUCKET_BITS 17
#define BUCKETS (1 << BUCKET_BITS)
#define MASK (BUCKETS - 1)

static inline uint32_t hash_ngram(int pos, const unsigned char *bytes, int order) {
    uint32_t h = pos;
    for (int i = 0; i < order; i++) {
        h = h * BASE + bytes[i];
    }
    return h & MASK;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int buckets = hd[1], C = hd[2], ngram_bitmap = hd[3];
    if (buckets != BUCKETS || C < 1 || C > 256) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16L + (long)buckets * C * 4L + (long)C * 4L;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *bias = W + (long)buckets * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        memcpy(scores, bias, sizeof(float) * C);

        for (int order = 1; order <= 3; order++) {
            if (!(ngram_bitmap & (1 << order))) continue;
            for (int p = 0; p <= 16 - order; p++) {
                uint32_t bucket = hash_ngram(p, xi + p, order);
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