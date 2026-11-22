```hypothesis
Family: label-hash (linear model on hashed unigram+bigram features).
Why: Language modeling benefits from local character interactions; bigrams capture pairwise dependencies that one-hot linear models miss. Feature hashing compresses the huge bigram space (16*256^2) into a fixed-size table (D=2048), acting as regularization and keeping inference fast (31 features * 65 classes ≈ 2k adds/item). Tuning D on the held-out 20% should yield >0.3375 accuracy at ~2x label-linear CPU cost.
Failure: dev accuracy ≤ 0.3375 or instructions/item > 15000.
```

```python
import json, math, struct, sys, hashlib

import numpy as np

# hyperparameters to tune on the tuning split (last 20% of train.in)
D_CANDIDATES = [1024, 2048, 4096, 8192]
LR_CANDIDATES = [1e-2, 3e-3, 1e-3]
L2 = 1e-5
BATCH = 128
MAX_EPOCHS = 50
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_feature(feat, D):
    # feat is a tuple of ints
    h = 0
    for x in feat:
        h = (h * 31 + x) & 0xFFFFFFFF
    # MurmurHash3 finalizer
    h ^= h >> 16
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= h >> 16
    return h % D

def extract_features(context, D):
    # context: 16 bytes
    feats = []
    # unigrams
    for j, b in enumerate(context):
        feats.append(hash_feature((1, j, b), D))
    # bigrams
    for j in range(15):
        feats.append(hash_feature((2, j, context[j], context[j+1]), D))
    return np.array(feats, dtype=np.int32)  # 31 features

def train_model(X, y, D, C, lr, epochs, rng):
    n = X.shape[0]
    W = np.zeros((D, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            batch_X = X[idx]
            batch_y = y[idx]
            # forward
            scores = np.zeros((len(idx), C), dtype=np.float32)
            for i, feats in enumerate(batch_X):
                scores[i] = b + W[feats].sum(axis=0)
            # softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(batch_y)), batch_y] -= 1.0
            grad = probs / len(batch_y)  # (B, C)
            # backward
            gW = np.zeros_like(W)
            for i, feats in enumerate(batch_X):
                gW[feats] += grad[i]
            gW += L2 * W
            gb = grad.sum(axis=0) + L2 * b
            # Adam update
            t += 1
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b

def evaluate(W, b, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        scores = b + W[X[i]].sum(axis=0)
        pred = scores.argmax()
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    X_recs = records(train_in)
    y_recs = records(train_key)
    X = np.array([list(r) for r in X_recs], dtype=np.uint8)  # (N, 16)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    C = card["classes"]
    N = X.shape[0]
    # split last 20% for tuning
    split = int(0.8 * N)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_D = D_CANDIDATES[0]
    best_lr = LR_CANDIDATES[0]
    best_W = None
    best_b = None

    # Precompute features for tuning set for each D
    tune_feats = {}
    for D in D_CANDIDATES:
        tune_feats[D] = np.array([extract_features(ctx, D) for ctx in X_tune])

    for D in D_CANDIDATES:
        train_feats = np.array([extract_features(ctx, D) for ctx in X_train])
        for lr in LR_CANDIDATES:
            W, b = train_model(train_feats, y_train, D, C, lr, min(MAX_EPOCHS, 10), rng)
            # quick eval on tuning
            acc = evaluate(W, b, tune_feats[D], y_tune)
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_lr = lr
                best_W, best_b = W.copy(), b.copy()
            print(f"D={D} lr={lr} tune_acc={acc:.4f}", file=sys.stderr)

    print(f"Best: D={best_D} lr={best_lr} acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data with best hyperparameters
    all_feats = np.array([extract_features(ctx, best_D) for ctx in X])
    rng_final = np.random.default_rng(SEED)
    final_W, final_b = train_model(all_feats, y, best_D, C, best_lr, MAX_EPOCHS, rng_final)

    # Write weights.bin
    # magic: 0x48415348 ("HASH")
    magic = 0x48415348
    out = struct.pack("<4i", magic, 16, best_D, C)  # d=16 (context bytes), D, C
    out += final_W.astype("<f4").tobytes()
    out += final_b.astype("<f4").tobytes()
    open(weights_out, "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-hash inference: linear model on hashed unigram+bigram features */
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

static uint32_t hash_feature(uint32_t h, uint32_t D) {
    // MurmurHash3 finalizer (same as Python)
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h % D;
}

static uint32_t hash_unigram(int pos, int byte, uint32_t D) {
    uint32_t h = 0;
    h = h * 31 + 1;       // type=1
    h = h * 31 + pos;
    h = h * 31 + byte;
    return hash_feature(h, D);
}

static uint32_t hash_bigram(int pos, int b1, int b2, uint32_t D) {
    uint32_t h = 0;
    h = h * 31 + 2;       // type=2
    h = h * 31 + pos;
    h = h * 31 + b1;
    h = h * 31 + b2;
    return hash_feature(h, D);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // init scores with bias
        for (int c = 0; c < C; c++) scores[c] = b[c];
        // unigrams
        for (int j = 0; j < 16; j++) {
            uint32_t idx = hash_unigram(j, xi[j], D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // bigrams
        for (int j = 0; j < 15; j++) {
            uint32_t idx = hash_bigram(j, xi[j], xi[j+1], D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```