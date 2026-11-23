```hypothesis
Family: label-hash (feature-hashed n-gram linear model).
Why: The task predicts the next character from 16 previous bytes. Label-linear uses only unigram features (one-hot per position), ignoring local character interactions. By adding bigram features (adjacent byte pairs) via feature hashing, we capture short-range dependencies crucial for language modeling. A linear model on hashed unigram+bigram features remains fast at inference (sparse dot product of ~31 features) and can be trained with the same Adam optimizer. The hash dimension D is tuned on the validation split (last 20% of train). If validation accuracy does not exceed label-linear, it indicates bigram interactions are not sufficiently predictive or hash collisions hurt performance.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (will be tuned on validation split)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Candidate hash dimensions to try
D_CANDIDATES = [16384, 32768, 65536, 131072]

# Hash function (must match infer.c)
MULT = 1315423911  # 32-bit prime, but we use 64-bit arithmetic
def hash_feature(tup, D):
    h = 0
    for x in tup:
        h = (h * MULT + x) & 0xFFFFFFFFFFFFFFFF
    return h % D

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_feature_indices(X, D):
    """X is (n, 16) uint8. Returns (n, 31) int32 feature indices."""
    n = X.shape[0]
    feat_idx = np.empty((n, 31), dtype=np.int32)
    for i in range(n):
        xi = X[i]
        idx = 0
        # unigrams
        for p in range(16):
            feat_idx[i, idx] = hash_feature((p, int(xi[p])), D)
            idx += 1
        # bigrams
        for p in range(15):
            feat_idx[i, idx] = hash_feature((p, int(xi[p]), int(xi[p+1])), D)
            idx += 1
    return feat_idx

def train_model(X, y, feat_idx, C, D, epochs, rng):
    n = X.shape[0]
    # Initialize weights
    W = (rng.normal(0, math.sqrt(2.0 / D), size=(D, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            fi = feat_idx[idx]  # (batch, 31)
            yy = y[idx]
            # Forward: scores = sum_{f in fi} W[f] + b
            batch_scores = np.sum(W[fi], axis=1) + b  # (batch, C)
            # Softmax
            max_s = batch_scores.max(axis=1, keepdims=True)
            exp_s = np.exp(batch_scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            # Gradient
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = probs / len(yy)  # (batch, C)
            # Gradients for W and b
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, fi.ravel(), np.repeat(grad_scores, 31, axis=0))
            grad_W += L2 * W
            grad_b = grad_scores.sum(axis=0)
            grads = [grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b

def evaluate(X, y, feat_idx, W, b):
    # Compute accuracy
    scores = np.sum(W[feat_idx], axis=1) + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for validation (tuning split)
    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_size))
    
    best_D = None
    best_acc = -1.0
    best_W = None
    best_b = None
    
    for D in D_CANDIDATES:
        print(f"Trying D={D}", file=sys.stderr)
        feat_train = compute_feature_indices(X_train, D)
        feat_val = compute_feature_indices(X_val, D)
        # Train
        W, b = train_model(X_train, y_train, feat_train, C, D, epochs, rng)
        acc = evaluate(X_val, y_val, feat_val, W, b)
        print(f"  val acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_W, best_b = W.copy(), b.copy()
    
    print(f"Best D={best_D} val acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data with best_D
    feat_all = compute_feature_indices(X, best_D)
    W_final, b_final = train_model(X, y, feat_all, C, best_D, epochs, rng)
    
    # Save weights
    # Magic: 0x48534148 ('HASH' little-endian)
    magic = 0x48534148
    out = struct.pack("<3i", magic, best_D, C)
    out += best_W.astype("<f4").tobytes()
    out += b_final.astype("<f4").tobytes()
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

// Hash function matching train.py
static uint32_t hash_feature(const uint8_t *feat, int len, uint32_t D) {
    uint64_t h = 0;
    for (int i = 0; i < len; i++) {
        h = h * 1315423911 + feat[i];
    }
    return h % D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x48534148) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t D = hd[1], C = hd[2];
    // Expected size: 12 + D*C*4 + C*4
    long expected = 12 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    const float *b = (const float *)((const char *)W + (long)D * C * 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    uint8_t feat[3];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        // Initialize scores with bias
        for (int c = 0; c < (int)C; c++) scores[c] = b[c];
        // Unigrams
        for (int p = 0; p < 16; p++) {
            feat[0] = p; feat[1] = xi[p];
            uint32_t idx = hash_feature(feat, 2, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // Bigrams
        for (int p = 0; p < 15; p++) {
            feat[0] = p; feat[1] = xi[p]; feat[2] = xi[p+1];
            uint32_t idx = hash_feature(feat, 3, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < (int)C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```