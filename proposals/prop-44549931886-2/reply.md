```hypothesis
Family: label-ngram-linear. A linear model over unigram, bigram, and trigram features (hashed to fixed buckets).
Why: Character-level language modeling benefits from local n-gram statistics; unigrams alone (label-linear) miss adjacency patterns. Hashing keeps the feature space small and inference fast.
Failure signal: If dev accuracy does not exceed label-linear (0.3375), it indicates that higher-order n-grams do not add enough predictive power for a linear model on this task.
```

```python
"""label-ngram-linear: linear model over unigram/bigram/trigram features for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning (last 20% of train.in)
GRID = [
    {"lr": 0.03, "l2": 1e-4, "epochs_tune": 10, "epochs_full": 50},
    {"lr": 0.1,  "l2": 1e-4, "epochs_tune": 10, "epochs_full": 50},
    {"lr": 0.01, "l2": 1e-3, "epochs_tune": 10, "epochs_full": 50},
]
SEED = 1
BATCH = 64
B_BUCKETS = 2048
T_BUCKETS = 2048
UNIGRAM_SIZE = 16 * 256
NUM_FEATURES = UNIGRAM_SIZE + B_BUCKETS + T_BUCKETS
MAGIC = 0x4E474C31  # "NGL1"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def get_features(x):
    """x: uint8 array of length 16. Returns list of feature indices."""
    feats = []
    # unigrams
    for p in range(16):
        feats.append(p * 256 + x[p])
    # bigrams
    for p in range(15):
        h = (x[p] * 256 + x[p+1]) % B_BUCKETS
        feats.append(UNIGRAM_SIZE + h)
    # trigrams
    for p in range(14):
        h = (x[p] * 65536 + x[p+1] * 256 + x[p+2]) % T_BUCKETS
        feats.append(UNIGRAM_SIZE + B_BUCKETS + h)
    return np.array(feats, dtype=np.int32)

def train_model(X_feats, y, C, lr, l2, epochs, rng):
    """Returns trained weight matrix (NUM_FEATURES, C)."""
    scale = math.sqrt(6.0 / NUM_FEATURES)
    W = rng.uniform(-scale, scale, (NUM_FEATURES, C)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            grad = np.zeros_like(W)
            for i in idx:
                feats = X_feats[i]
                logits = np.sum(W[feats], axis=0)
                logits -= logits.max()
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum()
                dlogits = probs
                dlogits[y[i]] -= 1.0
                for f in feats:
                    grad[f] += dlogits
            grad /= len(idx)
            grad += l2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W

def accuracy(X_feats, y, W):
    correct = 0
    for i in range(len(y)):
        logits = np.sum(W[X_feats[i]], axis=0)
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    C = card["classes"]
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    # Precompute features
    X_feats = [get_features(X_bytes[i]) for i in range(n)]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    rng = np.random.default_rng(SEED)
    # Tune hyperparameters
    best_acc = -1.0
    best_cfg = None
    for cfg in GRID:
        rng_tune = np.random.default_rng(SEED)
        W_tune = train_model(X_feats[:split], y[:split], C,
                             cfg["lr"], cfg["l2"], cfg["epochs_tune"], rng_tune)
        acc = accuracy(X_feats[split:], y[split:], W_tune)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
    # Retrain on all data with best config
    rng_full = np.random.default_rng(SEED)
    W_final = train_model(X_feats, y, C,
                          best_cfg["lr"], best_cfg["l2"], best_cfg["epochs_full"], rng_full)
    # Write weights.bin
    header = struct.pack("<4i", MAGIC, NUM_FEATURES, C, B_BUCKETS, T_BUCKETS)
    # Note: we pack 5 ints but header format expects 4? Actually we need 5: magic, num_features, C, B, T.
    # We'll pack as 5 ints and adjust infer.c accordingly.
    header = struct.pack("<5i", MAGIC, NUM_FEATURES, C, B_BUCKETS, T_BUCKETS)
    weights_bytes = W_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + weights_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram-linear inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4E474C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int num_features = hd[1], C = hd[2], B = hd[3], T = hd[4];
    const int UNIGRAM_SIZE = 16 * 256;
    if (C < 1 || C > 256 || num_features != UNIGRAM_SIZE + B + T) {
        fprintf(stderr, "bad weights dimensions\n"); return 3;
    }
    long expected = 20 + (long)num_features * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 5);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memset(logits, 0, sizeof(float) * C);
        // unigrams
        for (int p = 0; p < 16; p++) {
            int feat = p * 256 + xi[p];
            const float *w = W + (long)feat * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        // bigrams
        for (int p = 0; p < 15; p++) {
            int h = (xi[p] * 256 + xi[p+1]) % B;
            int feat = UNIGRAM_SIZE + h;
            const float *w = W + (long)feat * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        // trigrams
        for (int p = 0; p < 14; p++) {
            int h = (xi[p] * 65536 + xi[p+1] * 256 + xi[p+2]) % T;
            int feat = UNIGRAM_SIZE + B + h;
            const float *w = W + (long)feat * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```