```hypothesis
Family: label-hash (hashed n-gram linear model). Uses unigrams, bigrams, and trigrams of the 16-byte context, hashed to a fixed bucket array, trained with SGD on cross-entropy. Captures local character interactions missed by label-linear (unigrams only) while staying lighter than label-gbdt. Expected to beat label-linear's 33.75% dev accuracy with similar or lower CPU cost. Failure if dev accuracy ≤ label-linear or inference cost exceeds label-linear significantly.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
BUCKETS_GRID = [4096, 8192, 16384]
LR_GRID = [0.1, 0.05, 0.01]
L2_GRID = [1e-5, 1e-4]
EPOCHS_TUNE = 10
EPOCHS_FULL = 15
BATCH = 64
N_GRAMS = 3  # unigrams, bigrams, trigrams

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fnv1a_32(bytes_list):
    h = 2166136261
    for b in bytes_list:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def compute_indices(X, B):
    """X: (n, 16) uint8 -> indices: (n, 45) uint32"""
    n = X.shape[0]
    indices = np.empty((n, 45), dtype=np.uint32)
    col = 0
    # unigrams
    for i in range(16):
        indices[:, col] = np.array([fnv1a_32([X[row, i]]) % B for row in range(n)], dtype=np.uint32)
        col += 1
    # bigrams
    for i in range(15):
        indices[:, col] = np.array([fnv1a_32([X[row, i], X[row, i+1]]) % B for row in range(n)], dtype=np.uint32)
        col += 1
    # trigrams
    for i in range(14):
        indices[:, col] = np.array([fnv1a_32([X[row, i], X[row, i+1], X[row, i+2]]) % B for row in range(n)], dtype=np.uint32)
        col += 1
    return indices

def train_eval(W, indices, y, C, lr, l2, epochs, rng, batch_size=BATCH, verbose=False):
    n = indices.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            idx_indices = indices[idx]  # (m, 45)
            yy = y[idx]
            m = len(yy)
            # scores: (m, C)
            scores = np.zeros((m, C), dtype=np.float32)
            flat_idx = idx_indices.ravel()
            # gather weights for each bucket
            w_gathered = W[flat_idx]  # (m*45, C)
            w_gathered = w_gathered.reshape(m, 45, C)
            scores = w_gathered.sum(axis=1)
            # softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            # gradient
            probs[np.arange(m), yy] -= 1.0
            probs /= m
            # accumulate gradient per bucket
            grad = np.zeros_like(W)
            # repeat probs for each bucket
            probs_rep = np.repeat(probs, 45, axis=0)  # (m*45, C)
            np.add.at(grad, flat_idx, probs_rep)
            # L2
            grad += l2 * W
            # update
            W -= lr * grad
    return W

def accuracy(W, indices, y, C):
    n = indices.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    flat_idx = indices.ravel()
    w_gathered = W[flat_idx].reshape(n, 45, C)
    scores = w_gathered.sum(axis=1)
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)

    best_acc = -1
    best_W = None
    best_params = None

    for B in BUCKETS_GRID:
        print(f"Trying B={B}", file=sys.stderr)
        train_indices = compute_indices(X_train, B)
        tune_indices = compute_indices(X_tune, B)
        for lr in LR_GRID:
            for l2 in L2_GRID:
                W = np.zeros((B, C), dtype=np.float32)
                W = train_eval(W, train_indices, y_train, C, lr, l2, EPOCHS_TUNE, rng)
                acc = accuracy(W, tune_indices, y_tune, C)
                print(f"  B={B} lr={lr} l2={l2} -> {acc:.4f}", file=sys.stderr)
                if acc > best_acc:
                    best_acc = acc
                    best_W = W.copy()
                    best_params = (B, lr, l2)

    print(f"Best: B={best_params[0]} lr={best_params[1]} l2={best_params[2]} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    B, lr, l2 = best_params
    all_indices = compute_indices(X, B)
    W = np.zeros((B, C), dtype=np.float32)
    W = train_eval(W, all_indices, y, C, lr, l2, EPOCHS_FULL, rng)

    # Write weights.bin: magic, d, B, C, then float32 matrix (B*C) row-major
    magic = 0x48534148  # 'HASH' little-endian
    out = struct.pack("<4i", magic, d, B, C)
    out += W.astype("<f4").tobytes()
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

static uint32_t fnv1a_32(const unsigned char *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48534148) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], B = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16 || nw != 16 + 4L * (long)B * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        // unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 1) % B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 2) % B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // trigrams
        for (int p = 0; p < 14; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 3) % B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```