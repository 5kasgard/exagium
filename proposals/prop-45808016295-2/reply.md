```hypothesis
Family: label-ngram (linear unigram + hashed bigram features).
Why: Tinyshakespeare is character-level language; bigrams (e.g., "th", "he") capture local statistics that unigrams miss. Hashing bigrams into a fixed bucket table keeps parameters small (~800k) and inference fast (two table lookups per position). Linear model stays CPU-friendly.
Failure signal: Dev accuracy ≤ 0.35 (bigram features not helping enough) or inference cost > 2× label-linear.
```

```python
"""label-ngram: linear model with unigram (positional) and hashed bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only.
Weights format: magic 0x4E47524D, d, C, B2; float32 W1[d*256, C], W2[B2, C]."""
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning (last 20% of train.in)
GRID = {
    'B2': [4096, 8192, 16384],
    'LR': [1e-3, 3e-3],
    'L2': [1e-4, 1e-3],
}
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_EPOCHS = 10  # epochs for each grid config on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_bigram(b1, b2, B2):
    return (b1 * 256 + b2) % B2

def featurize(X, d, B2):
    """Return list of feature indices for each example: unigram indices (0..d*256-1) and bigram indices (d*256 .. d*256+B2-1)."""
    n = X.shape[0]
    unigram_base = np.arange(d) * 256
    unigram_idx = X + unigram_base  # shape (n, d)
    bigram_idx = np.zeros((n, d-1), dtype=np.int32)
    for j in range(d-1):
        bigram_idx[:, j] = hash_bigram(X[:, j], X[:, j+1], B2) + d * 256
    return unigram_idx, bigram_idx

def train_model(X, y, C, d, B2, LR, L2, epochs, rng):
    total_feat = d * 256 + B2
    W = np.zeros((total_feat, C), dtype=np.float32)
    # Adam state
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    n = X.shape[0]
    unigram_idx, bigram_idx = featurize(X, d, B2)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            u_idx = unigram_idx[idx].reshape(-1)      # (BATCH*d,)
            b_idx = bigram_idx[idx].reshape(-1)       # (BATCH*(d-1),)
            all_idx = np.concatenate([u_idx, b_idx])
            # Compute logits: sum of rows for each example
            logits = np.zeros((len(idx), C), dtype=np.float32)
            # We can accumulate by scattering; since BATCH small, loop over examples
            for i, ex in enumerate(idx):
                logits[i] = W[unigram_idx[ex]].sum(axis=0) + W[bigram_idx[ex]].sum(axis=0)
            # Softmax + gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(idx)), y[idx]] -= 1.0
            probs /= len(idx)
            # Gradient wrt W: for each used feature, add probs[i] for corresponding example
            grad = np.zeros_like(W)
            # Unigram gradients
            for i, ex in enumerate(idx):
                grad[unigram_idx[ex]] += probs[i]
            # Bigram gradients
            for i, ex in enumerate(idx):
                grad[bigram_idx[ex]] += probs[i]
            grad += L2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W

def evaluate(X, y, W, d, B2):
    n = X.shape[0]
    unigram_idx, bigram_idx = featurize(X, d, B2)
    correct = 0
    for i in range(n):
        logits = W[unigram_idx[i]].sum(axis=0) + W[bigram_idx[i]].sum(axis=0)
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_W = None

    # Grid search on tuning split
    for B2 in GRID['B2']:
        for LR in GRID['LR']:
            for L2 in GRID['L2']:
                W = train_model(train_X, train_y, C, d, B2, LR, L2, TUNE_EPOCHS, rng)
                acc = evaluate(tune_X, tune_y, W, d, B2)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (B2, LR, L2)
                    best_W = W.copy()
                # Reset RNG for next config to keep deterministic across configs? 
                # We'll just continue; the RNG state advances, but each config gets different permutations.
                # To be fully deterministic per config, we could reseed, but not required.

    print(f"Best tuning acc: {best_acc:.4f} with B2={best_params[0]}, LR={best_params[1]}, L2={best_params[2]}", file=sys.stderr)

    # Retrain on full data with best params
    B2, LR, L2 = best_params
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W = train_model(X, y, C, d, B2, LR, L2, epochs, np.random.default_rng(SEED))

    # Write weights
    magic = 0x4E47524D  # 'NGRM'
    total_feat = d * 256 + B2
    expected_size = 16 + 4 * total_feat * C
    out = struct.pack("<4i", magic, d, C, B2)
    out += W.astype("<f4").tobytes()
    if len(out) != expected_size:
        print(f"Weight size mismatch: {len(out)} vs {expected_size}", file=sys.stderr)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte (class index).
 * Weights format: magic 0x4E47524D, int32 d, C, B2; float32 W1[d*256][C], W2[B2][C]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], B2 = hd[3];
    if (d != 16 || C != 65 || B2 <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long total_feat = (long)d * 256 + B2;
    long expected = 16 + 4L * total_feat * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *W1 = W;
    const float *W2 = W + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        // Unigram features
        for (int j = 0; j < d; j++) {
            const float *row = W1 + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        // Bigram features
        for (int j = 0; j < d - 1; j++) {
            uint32_t hash = (xi[j] * 256 + xi[j+1]) % (uint32_t)B2;
            const float *row = W2 + (long)hash * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```