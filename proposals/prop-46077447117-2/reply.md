```hypothesis
Propose a "hashed n-gram linear" model (label-hngram). It maps all character n-grams (lengths 1–3) from the 16-byte context into a fixed-size hash table, then learns a linear classifier on the summed hash buckets. This captures local sequential interactions (bigrams/trigrams) without position-specific parameters, which is crucial for language modeling. It should beat label-linear (0.3375) by modeling character transitions. If it fails to beat label-linear, it indicates that absolute position information is more important than local n-gram statistics for this task, or that hash collisions degrade the signal.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains a linear model on hashed n-gram features (lengths 1,2,3) with Adam.
Hyperparameters chosen on the last 20% of train.in (tuning split), then retrain on all data.
Weights format: magic 0x484E474C, hash_buckets, max_ngram, classes, input_len,
float32 weight matrix [hash_buckets, classes], float32 bias [classes]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
HASH_BUCKETS_OPTS = [4096, 8192]
LR_OPTS = [0.01, 0.03, 0.1]
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
SEED = 1
MAX_NGRAM = 3  # unigrams, bigrams, trigrams

def fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash."""
    h = 0x811c9dc5
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

def extract_ngrams(seq, max_len):
    """Return list of n-gram byte strings from a byte sequence."""
    n = len(seq)
    grams = []
    for l in range(1, max_len + 1):
        for i in range(n - l + 1):
            grams.append(seq[i:i+l])
    return grams

def hash_ngrams(grams, buckets):
    """Hash list of n-grams to bucket indices."""
    return [fnv1a_32(g) % buckets for g in grams]

def sgd_step(z_indices, y, W, b, lr, l2, mW, vW, mb, vb, t, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam update for a minibatch. z_indices: list of lists of bucket indices per example."""
    n = len(z_indices)
    C = W.shape[1]
    # Compute logits and softmax
    logits = np.zeros((n, C), dtype=np.float32)
    for i, idxs in enumerate(z_indices):
        logits[i] = b + W[idxs].sum(axis=0)
    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # Gradients
    grad_W = np.zeros_like(W)
    grad_b = probs.sum(axis=0)
    for i, idxs in enumerate(z_indices):
        for idx in idxs:
            grad_W[idx] += probs[i]
    grad_W += l2 * W
    # Adam update
    t += 1
    mW *= beta1; mW += (1 - beta1) * grad_W
    vW *= beta2; vW += (1 - beta2) * (grad_W * grad_W)
    mb *= beta1; mb += (1 - beta1) * grad_b
    vb *= beta2; vb += (1 - beta2) * (grad_b * grad_b)
    mW_hat = mW / (1 - beta1 ** t)
    vW_hat = vW / (1 - beta2 ** t)
    mb_hat = mb / (1 - beta1 ** t)
    vb_hat = vb / (1 - beta2 ** t)
    W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
    b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    return t

def evaluate(z_indices_list, y, W, b):
    """Compute accuracy on a list of (indices, label)."""
    correct = 0
    for idxs, yy in zip(z_indices_list, y):
        scores = b + W[idxs].sum(axis=0)
        pred = np.argmax(scores)
        if pred == yy:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.array([list(r) for r in X_recs], dtype=np.uint8)  # (N, 16)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)    # (N,)
    N, d = X.shape
    C = card["classes"]
    assert d == card["input"]["bytes"] == 16
    assert C == 65

    # Split into train (first 80%) and tune (last 20%)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Precompute n-gram hashes for all examples for each hash bucket size
    # We'll do it on the fly during training to save memory, but for tuning we can precompute.
    # Since N is small (24000), we can precompute for each bucket size.
    rng = np.random.default_rng(SEED)

    best_acc = -1
    best_params = None
    best_W = None
    best_b = None

    for buckets in HASH_BUCKETS_OPTS:
        # Precompute hashed n-gram indices for train and tune
        def compute_all_indices(X_data):
            all_idxs = []
            for seq in X_data:
                grams = extract_ngrams(bytes(seq), MAX_NGRAM)
                idxs = hash_ngrams(grams, buckets)
                all_idxs.append(idxs)
            return all_idxs

        train_idxs = compute_all_indices(X_train)
        tune_idxs = compute_all_indices(X_tune)

        for lr in LR_OPTS:
            # Initialize weights
            W = (rng.uniform(-1, 1, (buckets, C)) * math.sqrt(6.0 / buckets)).astype(np.float32)
            b = np.zeros(C, dtype=np.float32)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            mb = np.zeros_like(b); vb = np.zeros_like(b)
            t = 0

            # Train on train split for TUNE_EPOCHS
            n_train = len(train_idxs)
            for epoch in range(TUNE_EPOCHS):
                order = rng.permutation(n_train)
                for s in range(0, n_train, BATCH):
                    batch_idx = order[s:s+BATCH]
                    z_batch = [train_idxs[i] for i in batch_idx]
                    y_batch = y_train[batch_idx]
                    t = sgd_step(z_batch, y_batch, W, b, lr, L2, mW, vW, mb, vb, t)

            # Evaluate on tune split
            acc = evaluate(tune_idxs, y_tune, W, b)
            if acc > best_acc:
                best_acc = acc
                best_params = (buckets, lr)
                best_W = W.copy()
                best_b = b.copy()
                # Save Adam state for the best config? We'll just retrain from scratch on all data.

    print(f"Best tuning: buckets={best_params[0]}, lr={best_params[1]}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best hyperparameters
    buckets, lr = best_params
    all_idxs = []
    for seq in X:
        grams = extract_ngrams(bytes(seq), MAX_NGRAM)
        idxs = hash_ngrams(grams, buckets)
        all_idxs.append(idxs)

    W = (rng.uniform(-1, 1, (buckets, C)) * math.sqrt(6.0 / buckets)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    n_all = len(all_idxs)
    for epoch in range(FINAL_EPOCHS):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            batch_idx = order[s:s+BATCH]
            z_batch = [all_idxs[i] for i in batch_idx]
            y_batch = y[batch_idx]
            t = sgd_step(z_batch, y_batch, W, b, lr, L2, mW, vW, mb, vb, t)

    # Write weights file
    magic = 0x484E474C  # "HNGL"
    header = struct.pack("<5i", magic, buckets, MAX_NGRAM, C, d)
    out = header + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (0-64).
 * Weights format: magic 0x484E474C, hash_buckets, max_ngram, classes, input_len,
 * float32 W[hash_buckets][classes], float32 bias[classes]. */
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

/* FNV-1a 32-bit hash, identical to Python version. */
static uint32_t fnv1a_32(const unsigned char *data, int len) {
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
    if (nw < 20 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights\n"); return 3; }
    int buckets = hd[1], max_ngram = hd[2], C = hd[3], d = hd[4];
    if (C < 1 || C > 256 || max_ngram != 3 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20 + 4L * buckets * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *bias = W + (long)buckets * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char *seq = malloc(d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(seq, xi, d);

        /* initialize scores with bias */
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        /* unigrams, bigrams, trigrams */
        for (int start = 0; start < d; start++) {
            for (int len = 1; len <= max_ngram && start + len <= d; len++) {
                uint32_t h = fnv1a_32(seq + start, len) % buckets;
                const float *w = W + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += w[c];
            }
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(seq); if (x) free(x);
    return 0;
}
```