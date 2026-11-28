```hypothesis
Family: label-hashngram (hashed n-gram linear model). Character-level language modeling benefits from local n-gram statistics. A linear model on hashed unigram, bigram, and trigram features captures pairwise and triple interactions without blowing up parameters. Hashing allows a fixed-size weight matrix (e.g., 8192 buckets) that fits in cache, making inference extremely fast (just a few hash computations and vector additions). This should beat label-linear (33.75%) by modeling local context interactions, while keeping CPU cost low. Failure: if dev accuracy ≤ 0.34 or instructions per item > 10000, the model does not improve over label-linear.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Features: unigrams (16), bigrams (15), trigrams (14) from the 16-byte context.
Each feature hashed to B buckets. Linear multinomial logistic regression trained with Adam.
Hyperparameters chosen on last 20% tuning split, then retrained on all data."""
import json, math, struct, sys
from array import array

import numpy as np

SEED = 1
RNG = np.random.default_rng(SEED)

# Fixed feature extraction: positions for unigrams, bigrams, trigrams
UNI_POS = list(range(16))
BI_POS = list(range(15))   # (i, i+1)
TRI_POS = list(range(14))  # (i, i+1, i+2)
N_FEAT = len(UNI_POS) + len(BI_POS) + len(TRI_POS)  # 45

# Hash function (FNV-1a 32-bit)
def fnv1a(bytes_list):
    h = 0x811c9dc5
    for b in bytes_list:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def compute_buckets(X, B):
    """X: (N, 16) uint8 -> buckets: (N, N_FEAT) int32"""
    N = X.shape[0]
    buckets = np.empty((N, N_FEAT), dtype=np.int32)
    col = 0
    # unigrams
    for i in UNI_POS:
        # feature: (i, byte)
        feat = (X[:, i].astype(np.uint32) + (i << 8))  # simple unique mapping per position
        # hash
        buckets[:, col] = np.array([fnv1a([i, b]) % B for b in X[:, i]], dtype=np.int32)
        col += 1
    # bigrams
    for i in BI_POS:
        buckets[:, col] = np.array([fnv1a([i, X[row, i], X[row, i+1]]) % B for row in range(N)], dtype=np.int32)
        col += 1
    # trigrams
    for i in TRI_POS:
        buckets[:, col] = np.array([fnv1a([i, X[row, i], X[row, i+1], X[row, i+2]]) % B for row in range(N)], dtype=np.int32)
        col += 1
    return buckets

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_loss_grad(logits, y):
    """logits: (batch, C), y: (batch,) int. Returns loss, grad (batch, C)"""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(len(y)), y] -= 1.0
    grad /= len(y)
    return loss, grad

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_sgd(X_buckets, y, B, C, lr, l2, batch, epochs, verbose=False):
    """Returns weight matrix W (B, C)"""
    W = RNG.normal(0, 0.01, size=(B, C)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    N = X_buckets.shape[0]
    for epoch in range(epochs):
        order = RNG.permutation(N)
        epoch_loss = 0.0
        for s in range(0, N, batch):
            idx = order[s:s+batch]
            buckets = X_buckets[idx]  # (batch, N_FEAT)
            yy = y[idx]
            # gather and sum
            logits = np.zeros((len(idx), C), dtype=np.float32)
            for f in range(N_FEAT):
                logits += W[buckets[:, f]]
            loss, grad = softmax_loss_grad(logits, yy)
            epoch_loss += loss * len(idx)
            # gradient wrt W: for each feature, add grad to corresponding bucket rows
            gW = np.zeros_like(W)
            for f in range(N_FEAT):
                np.add.at(gW, buckets[:, f], grad)
            gW += l2 * W
            t += 1
            m = beta1 * m + (1 - beta1) * gW
            v = beta2 * v + (1 - beta2) * (gW * gW)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose:
            print(f"epoch {epoch+1}: loss={epoch_loss/N:.4f}")
    return W

def evaluate(W, X_buckets, y):
    logits = np.zeros((X_buckets.shape[0], W.shape[1]), dtype=np.float32)
    for f in range(N_FEAT):
        logits += W[X_buckets[:, f]]
    return accuracy(logits, y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    N = X.shape[0]
    tune_size = N // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    # Hyperparameter grid (chosen on tuning split)
    grid = [
        {"B": 8192, "lr": 0.1, "l2": 1e-4, "batch": 64, "epochs": 20},
        {"B": 8192, "lr": 0.05, "l2": 1e-4, "batch": 64, "epochs": 20},
        {"B": 8192, "lr": 0.02, "l2": 1e-4, "batch": 64, "epochs": 20},
        {"B": 16384, "lr": 0.1, "l2": 1e-4, "batch": 64, "epochs": 20},
        {"B": 16384, "lr": 0.05, "l2": 1e-4, "batch": 64, "epochs": 20},
    ]
    best_acc = -1
    best_cfg = None
    best_W = None
    for cfg in grid:
        B = cfg["B"]
        train_buckets = compute_buckets(train_X, B)
        tune_buckets = compute_buckets(tune_X, B)
        W = train_sgd(train_buckets, train_y, B, C, cfg["lr"], cfg["l2"], cfg["batch"], cfg["epochs"])
        acc = evaluate(W, tune_buckets, tune_y)
        print(f"Config {cfg}: tune acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_W = W
    print(f"Best config: {best_cfg}, acc={best_acc:.4f}")

    # Retrain on full data with best config (more epochs)
    B = best_cfg["B"]
    full_buckets = compute_buckets(X, B)
    final_W = train_sgd(full_buckets, y, B, C, best_cfg["lr"], best_cfg["l2"], best_cfg["batch"], 40)

    # Save weights.bin
    # Header: magic (0x48415348), d, B, C
    magic = 0x48415348  # "HASH"
    header = struct.pack("<4i", magic, d, B, C)
    W_bytes = final_W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + W_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model inference.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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

/* FNV-1a 32-bit hash */
static uint32_t fnv1a(const unsigned char *data, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], B = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || nw != 16 + 4L * B * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    unsigned char feat_bytes[4];  // max 4 bytes for hash input (pos + up to 3 bytes)

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) logits[c] = 0.0f;

        // unigrams
        for (int p = 0; p < 16; p++) {
            feat_bytes[0] = p; feat_bytes[1] = xi[p];
            uint32_t bucket = fnv1a(feat_bytes, 2) % B;
            const float *wrow = W + (long)bucket * C;
            for (int c = 0; c < C; c++) logits[c] += wrow[c];
        }
        // bigrams
        for (int p = 0; p < 15; p++) {
            feat_bytes[0] = p; feat_bytes[1] = xi[p]; feat_bytes[2] = xi[p+1];
            uint32_t bucket = fnv1a(feat_bytes, 3) % B;
            const float *wrow = W + (long)bucket * C;
            for (int c = 0; c < C; c++) logits[c] += wrow[c];
        }
        // trigrams
        for (int p = 0; p < 14; p++) {
            feat_bytes[0] = p; feat_bytes[1] = xi[p]; feat_bytes[2] = xi[p+1]; feat_bytes[3] = xi[p+2];
            uint32_t bucket = fnv1a(feat_bytes, 4) % B;
            const float *wrow = W + (long)bucket * C;
            for (int c = 0; c < C; c++) logits[c] += wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```