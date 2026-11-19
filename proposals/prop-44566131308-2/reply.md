```hypothesis
Family: label-hashed-ngram (linear model on hashed unigram+bigram features).
Why: Tinyshakespeare has strong local character interactions (bigrams) that a pure unigram linear model misses. Hashing bigrams into a fixed-width table captures these interactions with minimal parameters and fast inference (31 feature lookups per example). A linear model on 2048 buckets should beat label-linear's 0.3375 accuracy while staying well under 10k instructions/item.
Failure signal: If dev accuracy ≤ 0.34 or instructions/item > 10000, the feature set or hash width is insufficient.
```
```python
"""label-hashed-ngram: linear model on hashed unigram+bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x484E4731, d=16, B=2048, C=65; float32 W[B][C], b[C]."""
import json, math, struct, sys

import numpy as np

B = 2048               # hash buckets
SEED = 1
LR, L2, BATCH, EPOCHS = 0.1, 1e-4, 64, 10
TUNE_FRAC = 0.2        # last 20% for tuning

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash32(x: int) -> int:
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = x ^ (x >> 16)
    return x & 0xffffffff

def featurize(X: np.ndarray) -> np.ndarray:
    """X: (n,16) uint8 -> (n,31) int32 bucket indices"""
    n = X.shape[0]
    buckets = np.empty((n, 31), dtype=np.int32)
    # 16 unigrams
    for pos in range(16):
        feat = (pos * 257 + X[:, pos].astype(np.int32) + 1)
        buckets[:, pos] = np.vectorize(lambda v: hash32(v) % B)(feat)
    # 15 bigrams
    for pos in range(15):
        feat = ((pos * 257 + X[:, pos].astype(np.int32)) * 257 + X[:, pos+1].astype(np.int32) + 1)
        buckets[:, 16+pos] = np.vectorize(lambda v: hash32(v) % B)(feat)
    return buckets

def softmax_cross_entropy_grad(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """scores: (batch, C), y: (batch,) -> gradient of loss w.r.t scores (batch, C)"""
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_one_split(X_tr, y_tr, X_te, y_te, lr, epochs):
    rng = np.random.default_rng(SEED)
    W = np.zeros((B, 65), dtype=np.float32)
    b = np.zeros(65, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = X_tr.shape[0]
    buckets_tr = featurize(X_tr)
    buckets_te = featurize(X_te)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            buck = buckets_tr[idx]
            yy = y_tr[idx]
            # forward
            scores = b + np.sum(W[buck], axis=1)   # (batch, C)
            # gradient
            g = softmax_cross_entropy_grad(scores, yy)   # (batch, C)
            # accumulate gradients for W
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, buck.reshape(-1), g.repeat(31, axis=0).reshape(-1, 65))
            grad_W += L2 * W
            grad_b = g.sum(axis=0) + L2 * b
            # Adam update
            t += 1
            for param, grad, m, v in [(W, grad_W, mW, vW), (b, grad_b, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # eval on tune
    scores_te = b + np.sum(W[buckets_te], axis=1)
    acc = (scores_te.argmax(axis=1) == y_te).mean()
    return W, b, acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert C == 65
    n = X.shape[0]
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # simple grid search on tuning split
    best_acc, best_W, best_b = -1, None, None
    for lr in [0.03, 0.1, 0.3]:
        for ep in [5, 10, 20]:
            W, b, acc = train_one_split(X_tr, y_tr, X_te, y_te, lr, ep)
            if acc > best_acc:
                best_acc, best_W, best_b = acc, W.copy(), b.copy()

    # retrain on full data with best lr/epochs (use same epochs as best)
    # we'll just run one more time on full data with the best lr and epochs=10 (fixed)
    rng = np.random.default_rng(SEED)
    W, b = best_W, best_b
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    buckets_all = featurize(X)
    for _ in range(10):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            buck = buckets_all[idx]
            yy = y[idx]
            scores = b + np.sum(W[buck], axis=1)
            g = softmax_cross_entropy_grad(scores, yy)
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, buck.reshape(-1), g.repeat(31, axis=0).reshape(-1, 65))
            grad_W += L2 * W
            grad_b = g.sum(axis=0) + L2 * b
            t += 1
            for param, grad, m, v in [(W, grad_W, mW, vW), (b, grad_b, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x484E4731  # "HNG1"
    out = struct.pack("<4i", magic, 16, B, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-hashed-ngram inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x484E4731, d=16, B=2048, C=65; float32 W[B][C], b[C]. */
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

static uint32_t hash32(uint32_t x) {
    x = (x ^ (x >> 16)) * 0x45d9f3b;
    x = (x ^ (x >> 16)) * 0x45d9f3b;
    x = x ^ (x >> 16);
    return x;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], B = hd[2], C = hd[3];
    if (d != 16 || B != 2048 || C != 65) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        // 16 unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t feat = (uint32_t)(p * 257 + xi[p] + 1);
            uint32_t bucket = hash32(feat) % B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // 15 bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t feat = ((uint32_t)(p * 257 + xi[p]) * 257 + xi[p+1] + 1);
            uint32_t bucket = hash32(feat) % B;
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