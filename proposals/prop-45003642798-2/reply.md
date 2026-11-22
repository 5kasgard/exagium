```hypothesis
Family: label-hash-ngram – linear model on hashed n-gram features (unigrams, bigrams, trigrams).
Why: Character prediction benefits from local context; hashing n-grams into a fixed feature space captures interactions without exploding parameters. Linear model on 45 features per example (16 uni + 15 bi + 14 tri) is fast (only additions) and should outperform independent unigrams (label-linear).
Failure: Dev accuracy ≤ 0.3375 (label-linear) or Ir/item ≥ 7359.9 (label-linear).
```

```python
"""label-hash-ngram: linear model on hashed n-gram features for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Feature hashing: 16 unigrams, 15 bigrams, 14 trigrams -> F buckets.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Tunes F on last 20% of train.in, then retrains on all data.
Weights format: magic 0x484E474C, int32 F, C; float32 W[F][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# Candidate feature space sizes to try on tuning split
F_CANDIDATES = [2048, 4096, 8192, 16384]
TUNE_EPOCHS = 10
FINAL_EPOCHS = 50

MAGIC = 0x484E474C  # 'HNGL'
PRIME = 0x9e3779b1  # 32-bit golden ratio prime
N_UNI = 16
N_BI = 15
N_TRI = 14
N_FEATS = N_UNI + N_BI + N_TRI  # 45

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_feat(pos, *bytes_):
    """Deterministic 32-bit hash for n-gram at position pos."""
    h = pos
    for b in bytes_:
        h = h * 257 + b
    return (h * PRIME) & 0xffffffff

def featurize(X, F):
    """X: (n, 16) uint8 -> indices: (n, 45) int32"""
    n = X.shape[0]
    idx = np.empty((n, N_FEATS), dtype=np.int32)
    # unigrams
    for p in range(N_UNI):
        idx[:, p] = (hash_feat(p, X[:, p]) % F).astype(np.int32)
    # bigrams
    for p in range(N_BI):
        idx[:, N_UNI + p] = (hash_feat(p, X[:, p], X[:, p+1]) % F).astype(np.int32)
    # trigrams
    for p in range(N_TRI):
        idx[:, N_UNI + N_BI + p] = (hash_feat(p, X[:, p], X[:, p+1], X[:, p+2]) % F).astype(np.int32)
    return idx

def train_eval(X_tr, y_tr, X_te, y_te, F, C, epochs, rng):
    """Train on (X_tr, y_tr), return accuracy on (X_te, y_te)."""
    n_tr = X_tr.shape[0]
    idx_tr = featurize(X_tr, F)
    idx_te = featurize(X_te, F)
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            batch_idx = order[s:s+BATCH]
            feat_idx = idx_tr[batch_idx]          # (b, 45)
            y_batch = y_tr[batch_idx]
            bsz = len(batch_idx)
            # forward: sum weights for each example
            logits = b + W[feat_idx].sum(axis=1)  # (bsz, C)
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), y_batch] -= 1.0
            probs /= bsz
            # gradients
            gW = np.zeros_like(W)
            gb = probs.sum(axis=0)
            flat_idx = feat_idx.ravel()
            rep_grad = np.repeat(probs, N_FEATS, axis=0)
            np.add.at(gW, flat_idx, rep_grad)
            gW += L2 * W
            # Adam update
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # eval
    logits_te = b + W[idx_te].sum(axis=1)
    preds = logits_te.argmax(axis=1)
    return (preds == y_te).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    # split: last 20% for tuning
    split = n - n // 5
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_F, best_acc = F_CANDIDATES[0], -1.0
    for F in F_CANDIDATES:
        acc = train_eval(X_tr, y_tr, X_te, y_te, F, C, TUNE_EPOCHS, np.random.default_rng(SEED))
        if acc > best_acc:
            best_acc, best_F = acc, F
    # final training on all data
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    # use a fresh RNG but same seed for reproducibility
    rng_final = np.random.default_rng(SEED)
    n_all = n
    idx_all = featurize(X, best_F)
    W = (rng_final.uniform(-1, 1, (best_F, C)) * math.sqrt(6.0 / best_F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(final_epochs):
        order = rng_final.permutation(n_all)
        for s in range(0, n_all, BATCH):
            batch_idx = order[s:s+BATCH]
            feat_idx = idx_all[batch_idx]
            y_batch = y[batch_idx]
            bsz = len(batch_idx)
            logits = b + W[feat_idx].sum(axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), y_batch] -= 1.0
            probs /= bsz
            gW = np.zeros_like(W)
            gb = probs.sum(axis=0)
            flat_idx = feat_idx.ravel()
            rep_grad = np.repeat(probs, N_FEATS, axis=0)
            np.add.at(gW, flat_idx, rep_grad)
            gW += L2 * W
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # write weights
    out = struct.pack("<3i", MAGIC, best_F, C)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-hash-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E474C, int32 F, C; float32 W[F][C], b[C].
 * Features: 16 unigrams, 15 bigrams, 14 trigrams (45 total), hashed to F buckets. */
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

static uint32_t hash_feat(int pos, uint32_t b1, uint32_t b2, uint32_t b3) {
    uint32_t h = pos;
    h = h * 257 + b1;
    if (b2 != 0xffffffff) h = h * 257 + b2;
    if (b3 != 0xffffffff) h = h * 257 + b3;
    return h * 0x9e3779b1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights\n"); return 3; }
    int F = hd[1], C = hd[2];
    long expected = 12 + 4L * F * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)F * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        uint32_t idx[45];
        int f = 0;
        // unigrams
        for (int p = 0; p < 16; p++)
            idx[f++] = hash_feat(p, xi[p], 0xffffffff, 0xffffffff) % F;
        // bigrams
        for (int p = 0; p < 15; p++)
            idx[f++] = hash_feat(p, xi[p], xi[p+1], 0xffffffff) % F;
        // trigrams
        for (int p = 0; p < 14; p++)
            idx[f++] = hash_feat(p, xi[p], xi[p+1], xi[p+2]) % F;
        for (int j = 0; j < 45; j++) {
            const float *w = W + (long)idx[j] * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```