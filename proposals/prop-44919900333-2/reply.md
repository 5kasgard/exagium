```hypothesis
Propose a hashed n-gram logistic regression (family: label-hashngram). It maps unigrams and bigrams from the 16-byte context into a fixed feature vector via a fast 64-bit hash, then learns a linear classifier with Adam. This captures local character interactions crucial for text prediction while keeping model size and inference cost low. If dev accuracy does not exceed label-linear (0.3375) or inference cost per item is not lower, the approach is ineffective.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning (last 20% of train)
D_GRID = [4096, 8192, 16384]
LR_GRID = [0.1, 0.03, 0.01]
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 3
FINAL_EPOCHS = 10
SEED = 1

# Hash constants
MASK64 = (1 << 64) - 1
UNIGRAM_SEED = 0x123456789abcdef0
BIGRAM_SEED  = 0xfedcba9876543210
MUL1 = 0x9e3779b97f4a7c15
MUL2 = 0xff51afd7ed558ccd
MUL3 = 0xc4ceb9fe1a85ec53

def mix64(h):
    h = (h ^ (h >> 33)) & MASK64
    h = (h * MUL2) & MASK64
    h = (h ^ (h >> 33)) & MASK64
    h = (h * MUL3) & MASK64
    h = (h ^ (h >> 33)) & MASK64
    return h

def hash_ngram(bytes_list, seed):
    h = seed
    for b in bytes_list:
        h = (h ^ b) & MASK64
        h = (h * MUL1) & MASK64
    return mix64(h)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_hashes(X):
    """X is (N, 16) uint8. Returns (N, 31) uint64 hashes for 16 unigrams + 15 bigrams."""
    N = X.shape[0]
    hashes = np.empty((N, 31), dtype=np.uint64)
    for n in range(N):
        x = X[n]
        idx = 0
        for pos in range(16):
            hashes[n, idx] = hash_ngram([pos, x[pos]], UNIGRAM_SEED)
            idx += 1
        for pos in range(15):
            hashes[n, idx] = hash_ngram([pos, x[pos], x[pos+1]], BIGRAM_SEED)
            idx += 1
    return hashes

def indices_signs(hashes, D):
    idx = hashes % D
    sgn = 1 - 2 * ((hashes >> 63) & 1).astype(np.int8)
    return idx.astype(np.int32), sgn.astype(np.int8)

def train_eval(D, lr, X_tr, y_tr, X_val, y_val, hashes_tr, hashes_val, C, epochs):
    idx_tr, sgn_tr = indices_signs(hashes_tr, D)
    idx_val, sgn_val = indices_signs(hashes_val, D)
    n_tr = X_tr.shape[0]
    n_val = X_val.shape[0]

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            batch_idx = order[s:s+BATCH]
            b_idx = idx_tr[batch_idx]
            b_sgn = sgn_tr[batch_idx]
            y_batch = y_tr[batch_idx]
            B = len(batch_idx)

            # forward
            logits = np.zeros((B, C), dtype=np.float32)
            for f in range(31):
                logits += b_sgn[:, f, None] * W[b_idx[:, f]]
            # softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # gradient
            probs[np.arange(B), y_batch] -= 1.0
            probs /= B
            grad = np.zeros_like(W)
            for f in range(31):
                np.add.at(grad, b_idx[:, f], b_sgn[:, f, None] * probs)
            grad += L2 * W

            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # evaluate on validation
    logits_val = np.zeros((n_val, C), dtype=np.float32)
    for f in range(31):
        logits_val += sgn_val[:, f, None] * W[idx_val[:, f]]
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, W

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    N = X.shape[0]
    split = int(0.8 * N)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # precompute hashes for all data
    hashes_tr = compute_hashes(X_tr)
    hashes_val = compute_hashes(X_val)

    best_acc = -1
    best_D = D_GRID[0]
    best_lr = LR_GRID[0]
    best_W = None

    for D in D_GRID:
        for lr in LR_GRID:
            acc, W = train_eval(D, lr, X_tr, y_tr, X_val, y_val, hashes_tr, hashes_val, C, TUNE_EPOCHS)
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_lr = lr
                best_W = W.copy()

    # retrain on all data with best hyperparameters
    hashes_all = compute_hashes(X)
    idx_all, sgn_all = indices_signs(hashes_all, best_D)
    n_all = N

    W = best_W
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    rng = np.random.default_rng(SEED)

    for _ in range(FINAL_EPOCHS):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            batch_idx = order[s:s+BATCH]
            b_idx = idx_all[batch_idx]
            b_sgn = sgn_all[batch_idx]
            y_batch = y[batch_idx]
            B = len(batch_idx)

            logits = np.zeros((B, C), dtype=np.float32)
            for f in range(31):
                logits += b_sgn[:, f, None] * W[b_idx[:, f]]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            probs /= B
            grad = np.zeros_like(W)
            for f in range(31):
                np.add.at(grad, b_idx[:, f], b_sgn[:, f, None] * probs)
            grad += L2 * W

            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= best_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # save weights
    magic = 0x484E474C  # "HNGL"
    out = struct.pack("<IIIII", magic, best_D, C, 0, 0)
    out += W.astype("<f4").tobytes()
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

// 64-bit hash constants
static const uint64_t MASK64 = 0xFFFFFFFFFFFFFFFFULL;
static const uint64_t UNIGRAM_SEED = 0x123456789abcdef0ULL;
static const uint64_t BIGRAM_SEED  = 0xfedcba9876543210ULL;
static const uint64_t MUL1 = 0x9e3779b97f4a7c15ULL;
static const uint64_t MUL2 = 0xff51afd7ed558ccdULL;
static const uint64_t MUL3 = 0xc4ceb9fe1a85ec53ULL;

static uint64_t mix64(uint64_t h) {
    h ^= h >> 33;
    h *= MUL2;
    h ^= h >> 33;
    h *= MUL3;
    h ^= h >> 33;
    return h;
}

static uint64_t hash_ngram(const unsigned char *bytes, int len, uint64_t seed) {
    uint64_t h = seed;
    for (int i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= MUL1;
    }
    return mix64(h);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights\n"); return 3; }
    int D = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 20 + 4L * (long)D * C) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 5);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char ngram_bytes[3];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memset(logits, 0, sizeof(float) * C);

        // 16 unigrams
        for (int p = 0; p < 16; p++) {
            ngram_bytes[0] = p;
            ngram_bytes[1] = xi[p];
            uint64_t h = hash_ngram(ngram_bytes, 2, UNIGRAM_SEED);
            int idx = h % D;
            int sgn = (h >> 63) ? -1 : 1;
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += sgn * wrow[c];
        }
        // 15 bigrams
        for (int p = 0; p < 15; p++) {
            ngram_bytes[0] = p;
            ngram_bytes[1] = xi[p];
            ngram_bytes[2] = xi[p+1];
            uint64_t h = hash_ngram(ngram_bytes, 3, BIGRAM_SEED);
            int idx = h % D;
            int sgn = (h >> 63) ? -1 : 1;
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += sgn * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```