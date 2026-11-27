```hypothesis
We propose label-ngram, a linear model over hashed n-grams (unigrams, bigrams, trigrams) of the 16-byte context.
This captures local character interactions crucial for language modeling while remaining a simple linear model with fast inference.
If it does not beat label-linear (0.3375) on dev, it would show that n-gram features alone are insufficient without deeper nonlinearities.
```

```python
"""label-ngram: linear model over hashed n-grams (1,2,3) of 16-byte context.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
GRID = {
    'B': [2048, 4096],
    'lr': [1e-3, 3e-3],
    'l2': [1e-4, 1e-3],
}
SEED = 1
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # max samples per epoch

FNV_OFFSET = 0x811c9dc5
FNV_PRIME = 0x01000193

def fnv1a_32(data, B):
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xffffffff
    return h % B

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_buckets(seq, B):
    """seq: 16 bytes. Return list of bucket indices for n-grams of order 1,2,3."""
    buckets = []
    # order 1
    for i in range(16):
        buckets.append(fnv1a_32(bytes([seq[i]]), B))
    # order 2
    for i in range(15):
        buckets.append(fnv1a_32(seq[i:i+2], B))
    # order 3
    for i in range(14):
        buckets.append(fnv1a_32(seq[i:i+3], B))
    return buckets

def train_model(X, y, C, B, lr, l2, epochs, rng):
    n = X.shape[0]
    # Precompute buckets for all examples
    all_buckets = [compute_buckets(X[i], B) for i in range(n)]
    # Initialize weights and bias
    W = (rng.uniform(-1, 1, (B, C)) * math.sqrt(6.0 / B)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            bs = len(idx)
            # Forward: scores = bias + sum W[bucket]
            scores = np.tile(bias, (bs, 1))
            for b_i, ex_idx in enumerate(idx):
                for bucket in all_buckets[ex_idx]:
                    scores[b_i] += W[bucket]
            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), y[idx]] -= 1.0
            grad_scores = probs / bs  # (bs, C)
            # Gradients
            gW = np.zeros_like(W)
            gb = grad_scores.sum(axis=0)
            for b_i, ex_idx in enumerate(idx):
                for bucket in all_buckets[ex_idx]:
                    gW[bucket] += grad_scores[b_i]
            gW += l2 * W
            # Adam update
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (bias, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, bias

def evaluate(X, y, C, B, W, bias):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        buckets = compute_buckets(X[i], B)
        scores = bias.copy()
        for bucket in buckets:
            scores += W[bucket]
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_W = None
    best_bias = None
    # Grid search
    for B in GRID['B']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
                W, bias = train_model(X_train, y_train, C, B, lr, l2, epochs, rng)
                acc = evaluate(X_val, y_val, C, B, W, bias)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (B, lr, l2, epochs)
                    best_W, best_bias = W.copy(), bias.copy()
    print(f"Best tuning acc: {best_acc:.4f} with B={best_params[0]}, lr={best_params[1]}, l2={best_params[2]}, epochs={best_params[3]}", file=sys.stderr)
    # Retrain on full data with best params
    B, lr, l2, epochs = best_params
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, bias = train_model(X, y, C, B, lr, l2, epochs_full, rng)
    # Save weights
    magic = 0x4C464D31  # 'LFM1'
    header = struct.pack("<4I", magic, B, C, d)
    out = header + bias.astype("<f4").tobytes() + W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: linear model over hashed n-grams (1,2,3) of 16-byte context.
 * ./infer weights.bin in.rec N
 * Output: N records, each 1 byte (class index). */
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

static uint32_t fnv1a_32(const unsigned char *data, size_t len, uint32_t B) {
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h % B;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t B = hd[1], C = hd[2], d = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * C + 4L * B * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *W = (const float *)(bias + C);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        // Initialize scores with bias
        for (uint32_t c = 0; c < C; c++) scores[c] = bias[c];
        // Unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 1, B);
            const float *w = W + (size_t)bucket * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w[c];
        }
        // Bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 2, B);
            const float *w = W + (size_t)bucket * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w[c];
        }
        // Trigrams
        for (int p = 0; p < 14; p++) {
            uint32_t bucket = fnv1a_32(xi + p, 3, B);
            const float *w = W + (size_t)bucket * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```