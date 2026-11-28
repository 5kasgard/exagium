```hypothesis
Family: label-ngram (hashed n-gram logistic regression).
Why: tinyshakespeare is character-level text; local n-grams (uni/bi/tri-grams) with positional hashing capture short-range dependencies that independent per-position one-hots miss, while staying linear-time and tiny. The feature space (45 n-grams × 16 positions) is hashed to a fixed width (tuned 4k–16k), giving a compact linear model with strong inductive bias for language.
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear.
```
```python
"""label-ngram: hashed n-gram logistic regression for fixed-byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Features: for each of the 16 context bytes, unigram at pos; bigram at pos,pos+1; trigram at pos,pos+1,pos+2.
Each n-gram is hashed with its start position to a fixed table of size H (tuned on last 20% of train).
Linear model (C classes) trained with Adam (lr=3e-3, L2=1e-4, batch=64, epochs min(100, ceil(600000/n))).
Weights file: magic 0x4E475241 ('NGRA'), int32 H, d=16, C; float32 W[H*C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
HASH_CANDIDATES = [4096, 8192, 16384]  # tuned on last 20% of train
MAGIC = 0x4E475241  # 'NGRA'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fnv1a_32(buf: bytes) -> int:
    h = 0x811c9dc5
    for b in buf:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def featurize(x: np.ndarray, H: int) -> np.ndarray:
    """x: (n, 16) uint8 -> (n, 45) int32 feature indices (0..H-1)"""
    n = x.shape[0]
    feat = np.zeros((n, 45), dtype=np.int32)
    col = 0
    # unigrams: 16
    for p in range(16):
        buf = np.empty((n, 2), dtype=np.uint8)
        buf[:, 0] = p
        buf[:, 1] = x[:, p]
        feat[:, col] = np.frombuffer(buf.tobytes(), dtype=np.uint8).reshape(n, 2).view(np.uint16)[:, 0]  # wrong, redo
    # Let's do it cleanly in Python loops for clarity and determinism; n ≤ 24000, 45 features → 1M hashes, fine.
    for i in range(n):
        xi = x[i]
        idx = 0
        for p in range(16):
            feat[i, idx] = fnv1a_32(bytes([p, xi[p]])) % H; idx += 1
        for p in range(15):
            feat[i, idx] = fnv1a_32(bytes([p, xi[p], xi[p+1]])) % H; idx += 1
        for p in range(14):
            feat[i, idx] = fnv1a_32(bytes([p, xi[p], xi[p+1], xi[p+2]])) % H; idx += 1
    return feat

def train_linear(X_idx: np.ndarray, y: np.ndarray, H: int, C: int, rng: np.random.Generator):
    """X_idx: (n, 45) int32 feature indices. Returns W (H, C), b (C)."""
    n = X_idx.shape[0]
    W = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_idx[idx], y[idx]
            # forward: scores = b + sum_{k} W[feat_k]
            scores = b + np.sum(W[Xb], axis=1)  # (batch, C)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad = probs / len(yb)  # (batch, C)
            # gradient for W: for each feature index, add grad
            gW = np.zeros_like(W)
            np.add.at(gW, Xb.ravel(), np.repeat(grad, 45, axis=0))
            gW += L2 * W
            gb = grad.sum(axis=0) + L2 * b
            t += 1
            # Adam update
            mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            W -= LR * (mW / (1 - 0.9**t)) / (np.sqrt(vW / (1 - 0.999**t)) + 1e-8)
            b -= LR * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
    return W, b

def accuracy(X_idx: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    scores = b + np.sum(W[X_idx], axis=1)
    return float(np.mean(np.argmax(scores, axis=1) == y))

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]; C = card["classes"]
    assert d == 16
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split last 20% for tuning
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_H, best_acc = HASH_CANDIDATES[0], -1.0
    for H in HASH_CANDIDATES:
        Xtr_idx = featurize(X_train, H)
        Xte_idx = featurize(X_tune, H)
        W, b = train_linear(Xtr_idx, y_train, H, C, rng)
        acc = accuracy(Xte_idx, y_tune, W, b)
        if acc > best_acc:
            best_acc, best_H = acc, H
    # retrain on all data with best_H
    Xall_idx = featurize(X, best_H)
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    W, b = train_linear(Xall_idx, y, best_H, C, rng)
    # write weights
    out = struct.pack("<4i", MAGIC, best_H, d, C)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4E475241, int32 H, d=16, C; float32 W[H*C], b[C].
 * Features: 45 n-grams (16 uni, 15 bi, 14 tri) hashed with FNV-1a (32-bit) modulo H. */
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

static uint32_t fnv1a_32(const unsigned char *buf, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E475241) { fprintf(stderr, "bad weights\n"); return 3; }
    int H = hd[1], d = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || nw != 16 + 4L * ((long)H * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4), *b = W + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char feat_buf[4];  // max n-gram length = 4 (pos + 3 bytes)
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, C * sizeof(float));
        // 16 unigrams
        for (int p = 0; p < 16; p++) {
            feat_buf[0] = p; feat_buf[1] = xi[p];
            uint32_t idx = fnv1a_32(feat_buf, 2) % (uint32_t)H;
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // 15 bigrams
        for (int p = 0; p < 15; p++) {
            feat_buf[0] = p; feat_buf[1] = xi[p]; feat_buf[2] = xi[p+1];
            uint32_t idx = fnv1a_32(feat_buf, 3) % (uint32_t)H;
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // 14 trigrams
        for (int p = 0; p < 14; p++) {
            feat_buf[0] = p; feat_buf[1] = xi[p]; feat_buf[2] = xi[p+1]; feat_buf[3] = xi[p+2];
            uint32_t idx = fnv1a_32(feat_buf, 4) % (uint32_t)H;
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