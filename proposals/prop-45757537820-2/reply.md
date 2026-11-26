```hypothesis
We propose a "label-ngram" family: extract all n-grams (lengths 1–4) from the 16-byte context, hash them into a fixed-size feature space, and train a linear classifier (logistic regression) with Adam. This captures local character interactions vital for next-character prediction while keeping inference fast via feature hashing. It should beat label-linear because n-grams model multi-byte patterns that one-hot per-position cannot. If it fails to beat label-linear, it may be due to excessive hash collisions or insufficient capacity.
```

```python
import json, math, struct, sys, os

import numpy as np

# Fixed hyperparameters (will be tuned on the last 20% of train.in)
SEED = 1
MAX_N = 4            # maximum n-gram length
NUM_FEATURES = 16384 # hash table size (power of two)
BATCH = 64
EPOCHS = 10
LR = 0.1
L2 = 1e-4
HASH_SEED = 0x9e3779b9  # arbitrary constant for FNV-1a

# FNV-1a 32-bit hash
def fnv1a_32(data, seed=0x811c9dc5):
    h = seed
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def extract_ngrams(seq, max_n):
    """Return list of (length, ngram_bytes) for all ngrams up to max_n."""
    ngrams = []
    L = len(seq)
    for n in range(1, max_n + 1):
        for i in range(L - n + 1):
            ngrams.append((n, seq[i:i+n]))
    return ngrams

def hash_ngrams(ngrams, num_features):
    """Hash each ngram to a feature index."""
    indices = []
    for length, gram in ngrams:
        # prepend length to differentiate ngrams of different lengths
        h = fnv1a_32(bytes([length]) + gram, HASH_SEED)
        indices.append(h % num_features)
    return indices

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    # Load all training data
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = len(X_recs)
    assert all(len(r) == d for r in X_recs)

    # Precompute ngram feature indices for every example
    all_features = []
    example_ptr = [0]
    for rec in X_recs:
        ngrams = extract_ngrams(rec, MAX_N)
        idxs = hash_ngrams(ngrams, NUM_FEATURES)
        all_features.extend(idxs)
        example_ptr.append(len(all_features))
    all_features = np.array(all_features, dtype=np.int32)
    example_ptr = np.array(example_ptr, dtype=np.int32)

    # Split: last 20% for tuning
    split = int(n * 0.8)
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    # Simple grid search on tuning set
    best_acc = -1.0
    best_lr = LR
    best_feats = NUM_FEATURES
    # We'll try a few learning rates and feature sizes
    for lr in [0.03, 0.1, 0.3]:
        for feats in [8192, 16384, 32768]:
            # Re-hash with new feature size (could precompute multiple but fine for small grid)
            # For speed, we just re-hash the tuning set only
            tune_features = []
            tune_ptr = [0]
            for i in tune_idx:
                ngrams = extract_ngrams(X_recs[i], MAX_N)
                idxs = hash_ngrams(ngrams, feats)
                tune_features.extend(idxs)
                tune_ptr.append(len(tune_features))
            tune_features = np.array(tune_features, dtype=np.int32)
            tune_ptr = np.array(tune_ptr, dtype=np.int32)

            # Initialize weights
            W = (rng.uniform(-1, 1, (feats, C)) * math.sqrt(6.0 / feats)).astype(np.float32)
            bias = np.zeros(C, np.float32)
            mW, vW = np.zeros_like(W), np.zeros_like(W)
            mb, vb = np.zeros_like(bias), np.zeros_like(bias)
            t = 0

            # Train for a few epochs on train split
            for epoch in range(5):
                order = rng.permutation(split)
                for s in range(0, split, BATCH):
                    idx = order[s:s+BATCH]
                    # Forward
                    logits = np.zeros((len(idx), C), dtype=np.float32)
                    for k, ex in enumerate(idx):
                        feats_idx = all_features[example_ptr[ex]:example_ptr[ex+1]]
                        logits[k] = bias + W[feats_idx].sum(axis=0)
                    # Softmax
                    max_logits = logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits - max_logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    # Gradient
                    grads = probs
                    grads[np.arange(len(idx)), y[idx]] -= 1.0
                    grads /= len(idx)
                    # Accumulate weight gradients
                    W_grad = np.zeros_like(W)
                    for k, ex in enumerate(idx):
                        feats_idx = all_features[example_ptr[ex]:example_ptr[ex+1]]
                        np.add.at(W_grad, feats_idx, grads[k])
                    W_grad += L2 * W
                    bias_grad = grads.sum(axis=0) + L2 * bias
                    # Adam update
                    t += 1
                    mW = 0.9 * mW + 0.1 * W_grad
                    vW = 0.999 * vW + 0.001 * W_grad * W_grad
                    mW_hat = mW / (1 - 0.9 ** t)
                    vW_hat = vW / (1 - 0.999 ** t)
                    W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
                    mb = 0.9 * mb + 0.1 * bias_grad
                    vb = 0.999 * vb + 0.001 * bias_grad * bias_grad
                    mb_hat = mb / (1 - 0.9 ** t)
                    vb_hat = vb / (1 - 0.999 ** t)
                    bias -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)

            # Evaluate on tuning set
            correct = 0
            for i, ex in enumerate(tune_idx):
                feats_idx = tune_features[tune_ptr[i]:tune_ptr[i+1]]
                scores = bias + W[feats_idx].sum(axis=0)
                pred = np.argmax(scores)
                if pred == y[ex]:
                    correct += 1
            acc = correct / len(tune_idx)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_feats = feats

    # Retrain on all data with best hyperparameters
    NUM_FEATURES = best_feats
    LR = best_lr

    # Recompute all features with best feature size
    all_features = []
    example_ptr = [0]
    for rec in X_recs:
        ngrams = extract_ngrams(rec, MAX_N)
        idxs = hash_ngrams(ngrams, NUM_FEATURES)
        all_features.extend(idxs)
        example_ptr.append(len(all_features))
    all_features = np.array(all_features, dtype=np.int32)
    example_ptr = np.array(example_ptr, dtype=np.int32)

    W = (rng.uniform(-1, 1, (NUM_FEATURES, C)) * math.sqrt(6.0 / NUM_FEATURES)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            logits = np.zeros((len(idx), C), dtype=np.float32)
            for k, ex in enumerate(idx):
                feats_idx = all_features[example_ptr[ex]:example_ptr[ex+1]]
                logits[k] = bias + W[feats_idx].sum(axis=0)
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grads = probs
            grads[np.arange(len(idx)), y[idx]] -= 1.0
            grads /= len(idx)
            W_grad = np.zeros_like(W)
            for k, ex in enumerate(idx):
                feats_idx = all_features[example_ptr[ex]:example_ptr[ex+1]]
                np.add.at(W_grad, feats_idx, grads[k])
            W_grad += L2 * W
            bias_grad = grads.sum(axis=0) + L2 * bias
            t += 1
            mW = 0.9 * mW + 0.1 * W_grad
            vW = 0.999 * vW + 0.001 * W_grad * W_grad
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            mb = 0.9 * mb + 0.1 * bias_grad
            vb = 0.999 * vb + 0.001 * bias_grad * bias_grad
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            bias -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights file
    # Magic: 'NGRM' = 0x4E47524D
    magic = 0x4E47524D
    out = struct.pack("<4i", magic, NUM_FEATURES, C, MAX_N)
    out += struct.pack("<I", HASH_SEED)
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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

// FNV-1a 32-bit
static uint32_t fnv1a_32(const unsigned char *data, size_t len, uint32_t seed) {
    uint32_t h = seed;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wbuf;
    if (hd[0] != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int num_features = hd[1];
    int C = hd[2];
    int max_n = hd[3];
    uint32_t hash_seed = *(uint32_t *)(hd + 4);
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    size_t expected = 20 + (size_t)num_features * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(wbuf + 20);
    const float *bias = W + (size_t)num_features * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char ngram[16 + 1]; // max length 16 + length byte

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // zero scores
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // generate all ngrams up to max_n
        for (int n = 1; n <= max_n; n++) {
            for (int start = 0; start <= 16 - n; start++) {
                ngram[0] = (unsigned char)n;
                memcpy(ngram + 1, xi + start, n);
                uint32_t h = fnv1a_32(ngram, n + 1, hash_seed);
                int idx = h % num_features;
                const float *wrow = W + (size_t)idx * C;
                for (int c = 0; c < C; c++) scores[c] += wrow[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```