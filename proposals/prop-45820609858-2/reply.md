```hypothesis
Family: label-hashing (hashed n-gram features + linear classifier)
Why: Character prediction benefits from local n-gram statistics. Hashing n-grams (lengths 1-4) with position into a fixed-size feature vector captures interactions that per-position one-hot (label-linear) misses, while staying fast and memory-efficient. The linear model on hashed features is a strong baseline for text, and the fixed hash size allows tuning the accuracy-cost tradeoff.
Failure indication: If dev accuracy <= 0.3375 (label-linear) or inference cost exceeds label-linear significantly.
```

```python
"""label-hashing: hashed n-gram features (lengths 1-4) with position, linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned on validation split)
SEED = 1
MAX_NGRAM = 4
HASH_SIZES = [2048, 4096, 8192, 16384]
LRS = [0.1, 0.03, 0.01]
L2 = 1e-4
BATCH = 64
SEARCH_EPOCHS = 10
FINAL_EPOCHS = 50
SAMPLES = 600000  # for epoch ceiling


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a_hash(seq: bytes) -> int:
    h = 2166136261
    for b in seq:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def compute_features(contexts, hash_size, max_ngram):
    """contexts: list of bytes(length 16). Returns (n, hash_size) float32 array."""
    n = len(contexts)
    feats = np.zeros((n, hash_size), dtype=np.float32)
    # precompute (start, length) pairs
    pairs = []
    for L in range(1, max_ngram + 1):
        for s in range(0, 16 - L + 1):
            pairs.append((s, L))
    for i, ctx in enumerate(contexts):
        for s, L in pairs:
            ngram = ctx[s:s + L]
            seq = bytes([s, L]) + ngram
            h = fnv1a_hash(seq)
            idx = h % hash_size
            feats[i, idx] += 1.0
    return feats


def train_model(X, y, val_X, val_y, hash_size, lr, epochs, rng):
    n, C = X.shape[0], 65
    # Initialize weights
    W = (rng.normal(0, 0.01, (hash_size, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_val_acc = 0.0
    best_W, best_b = W.copy(), b.copy()

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            # forward
            logits = xb @ W + b  # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # loss gradient
            grad = probs
            grad[np.arange(len(yb)), yb] -= 1.0
            grad /= len(yb)
            # gradients
            dW = xb.T @ grad + L2 * W
            db = grad.sum(axis=0)
            # Adam update
            t += 1
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW * dW)
            mb = beta1 * mb + (1 - beta1) * db
            vb = beta2 * vb + (1 - beta2) * (db * db)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        # validation accuracy
        val_logits = val_X @ W + b
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == val_y).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_W, best_b = W.copy(), b.copy()
    return best_W, best_b, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]
    # map target byte -> class index
    byte_to_class = {ord(ch): i for i, ch in enumerate(class_chars)}

    train_in = records(sys.argv[2])
    train_key = records(sys.argv[3])
    X_all = [rec for rec in train_in]  # each 16 bytes
    y_all = np.array([byte_to_class[rec[0]] for rec in train_key], dtype=np.int64)

    # split: last 20% for validation (tuning split)
    n = len(X_all)
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # hyperparameter search
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_hp = None
    best_W = best_b = None

    for hs in HASH_SIZES:
        # compute features for train and val once per hash_size
        X_train_feat = compute_features(X_train, hs, MAX_NGRAM)
        X_val_feat = compute_features(X_val, hs, MAX_NGRAM)
        for lr in LRS:
            # fresh RNG for each run to keep deterministic across hp combos
            hp_rng = np.random.default_rng(SEED)
            W, b, val_acc = train_model(X_train_feat, y_train, X_val_feat, y_val, hs, lr, SEARCH_EPOCHS, hp_rng)
            if val_acc > best_acc:
                best_acc = val_acc
                best_hp = (hs, lr)
                best_W, best_b = W, b
    print(f"Best hp: hash_size={best_hp[0]}, lr={best_hp[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on all data with best hyperparameters
    hs, lr = best_hp
    X_all_feat = compute_features(X_all, hs, MAX_NGRAM)
    final_rng = np.random.default_rng(SEED)
    W, b, _ = train_model(X_all_feat, y_all, X_all_feat, y_all, hs, lr, FINAL_EPOCHS, final_rng)

    # write weights.bin
    # magic: b'HASH' (0x48415348)
    magic = 0x48415348
    header = struct.pack("<4i", magic, hs, MAX_NGRAM, C)
    # W is (hs, C) row-major float32
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(W_bytes)
        f.write(b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-hashing infer: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index (0-64). */
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

static uint32_t fnv1a_hash(const unsigned char *data, int len) {
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
    if (N == 0) return 0;  // start-up only

    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int hash_size = hd[1], max_ngram = hd[2], C = hd[3];
    if (C != 65 || max_ngram != 4 || hash_size <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * ((long)hash_size * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)hash_size * C;

    unsigned char *x = slurp(argv[2], &nx);
    float *features = (float *)malloc(hash_size * sizeof(float));
    float *scores = (float *)malloc(C * sizeof(float));
    unsigned char out_byte;

    // precompute (start, length) pairs
    int pairs_count = 0;
    int starts[58], lengths[58];
    for (int L = 1; L <= max_ngram; L++) {
        for (int s = 0; s <= 16 - L; s++) {
            starts[pairs_count] = s;
            lengths[pairs_count] = L;
            pairs_count++;
        }
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        memset(features, 0, hash_size * sizeof(float));
        for (int p = 0; p < pairs_count; p++) {
            int s = starts[p], L = lengths[p];
            unsigned char seq[2 + 4]; // max ngram length 4
            seq[0] = (unsigned char)s;
            seq[1] = (unsigned char)L;
            memcpy(seq + 2, xi + s, L);
            uint32_t h = fnv1a_hash(seq, 2 + L);
            int idx = h % hash_size;
            features[idx] += 1.0f;
        }

        // compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < hash_size; f++) {
            float fv = features[f];
            if (fv == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += fv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }

    free(features);
    free(scores);
    free(x);
    return 0;
}
```