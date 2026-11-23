```hypothesis
We propose a "label-ngram" family: a trigram language model with additive smoothing and linear interpolation of unigram, bigram, and trigram probabilities. It exploits the sequential nature of text and requires only the last two context bytes, making inference extremely fast (O(1) table lookups). Hyperparameters (smoothing alpha, interpolation weights) are tuned on the held-out 20% tuning split. If dev accuracy does not exceed label-linear's 0.3375, the hypothesis fails.
```

```python
"""label-ngram: trigram model with interpolation for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, uses only stdlib and numpy."""
import json, math, struct, sys
import numpy as np

SEED = 1
V = 65  # number of classes (from card)
ORDER = 3  # trigram

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_mapping(class_chars):
    """Map byte value -> class index (0..V-1), 255 for unknown."""
    mapping = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        mapping[ord(ch)] = idx
    return mapping

def count_ngrams(X, y, mapping):
    """Count unigram, bigram, trigram from data.
    X: (n, 16) uint8 context bytes
    y: (n,) uint8 target class indices (0..V-1)
    Returns counts_u (V,), counts_b (V, V), counts_t (V, V, V) as int64."""
    n = X.shape[0]
    # Map context bytes to class indices
    X_idx = mapping[X]  # (n, 16) with values 0..V-1 or 255
    # We assume no unknown bytes in training data (255 not present)
    # Use last two context bytes for trigram
    c1 = X_idx[:, -2]   # second last
    c2 = X_idx[:, -1]   # last
    t  = y              # target
    # Count unigram
    counts_u = np.bincount(t, minlength=V).astype(np.int64)
    # Count bigram: (prev, target)
    counts_b = np.zeros((V, V), dtype=np.int64)
    np.add.at(counts_b, (c2, t), 1)
    # Count trigram: (prev2, prev1, target)
    counts_t = np.zeros((V, V, V), dtype=np.int64)
    np.add.at(counts_t, (c1, c2, t), 1)
    return counts_u, counts_b, counts_t

def smooth_probs(counts_u, counts_b, counts_t, alpha):
    """Compute smoothed probability tables."""
    V = counts_u.shape[0]
    # Unigram
    total_u = counts_u.sum()
    p_u = (counts_u + alpha) / (total_u + alpha * V)
    # Bigram
    denom_b = counts_b.sum(axis=1, keepdims=True) + alpha * V
    p_b = (counts_b + alpha) / denom_b
    # Trigram
    denom_t = counts_t.sum(axis=2, keepdims=True) + alpha * V
    p_t = (counts_t + alpha) / denom_t
    return p_u.astype(np.float32), p_b.astype(np.float32), p_t.astype(np.float32)

def eval_accuracy(X, y, mapping, p_u, p_b, p_t, lambdas):
    """Compute accuracy on given data."""
    n = X.shape[0]
    X_idx = mapping[X]
    c1 = X_idx[:, -2]
    c2 = X_idx[:, -1]
    # Handle unknowns (should not happen in clean data)
    unk = (c1 == 255) | (c2 == 255)
    # Combined probabilities
    # For each example, we need p = lambda_t * p_t[c1,c2] + lambda_b * p_b[c2] + lambda_u * p_u
    # We'll compute for all classes and take argmax.
    # Vectorized: for each example, we can gather rows.
    # Since V=65, we can loop over examples for simplicity (n <= 24000).
    correct = 0
    lambda_t, lambda_b, lambda_u = lambdas
    for i in range(n):
        if unk[i]:
            probs = p_u
        else:
            probs = (lambda_t * p_t[c1[i], c2[i]] +
                     lambda_b * p_b[c2[i]] +
                     lambda_u * p_u)
        pred = np.argmax(probs)
        if pred == y[i]:
            correct += 1
    return correct / n

def tune_hyperparams(X_train, y_train, X_tune, y_tune, mapping):
    """Grid search over alpha and lambdas."""
    best_acc = -1
    best_alpha = 0.1
    best_lambdas = (1.0, 0.0, 0.0)
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    # Grid over lambdas (lambda_t, lambda_b, lambda_u) summing to 1
    # We'll step by 0.1
    for alpha in alphas:
        counts_u, counts_b, counts_t = count_ngrams(X_train, y_train, mapping)
        p_u, p_b, p_t = smooth_probs(counts_u, counts_b, counts_t, alpha)
        for lt in np.arange(0.0, 1.01, 0.1):
            for lb in np.arange(0.0, 1.01 - lt, 0.1):
                lu = 1.0 - lt - lb
                lambdas = (lt, lb, lu)
                acc = eval_accuracy(X_tune, y_tune, mapping, p_u, p_b, p_t, lambdas)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha
                    best_lambdas = lambdas
    return best_alpha, best_lambdas

def main():
    card = json.load(open(sys.argv[1]))
    # Input: 16 bytes
    d = card["input"]["bytes"]
    class_chars = card["class_chars"]
    V = card["classes"]
    assert V == 65
    # Read data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    # Mapping from byte to class index
    mapping = build_mapping(class_chars)
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    # Tune hyperparameters
    print(f"Tuning on {len(X_tune)} examples...", file=sys.stderr)
    alpha, lambdas = tune_hyperparams(X_train, y_train, X_tune, y_tune, mapping)
    print(f"Best alpha={alpha}, lambdas={lambdas}", file=sys.stderr)
    # Retrain on full data
    counts_u, counts_b, counts_t = count_ngrams(X, y, mapping)
    p_u, p_b, p_t = smooth_probs(counts_u, counts_b, counts_t, alpha)
    # Write weights.bin
    # Format:
    #   int32 magic (0x4E47524D)
    #   int32 version (1)
    #   int32 d (16)
    #   int32 V (65)
    #   int32 order (3)
    #   uint8 mapping[256]
    #   float32 unigram[V]
    #   float32 bigram[V*V]
    #   float32 trigram[V*V*V]
    #   float32 lambda[3]
    magic = 0x4E47524D
    version = 1
    order = 3
    header = struct.pack("<5i", magic, version, d, V, order)
    mapping_bytes = mapping.tobytes()
    unigram_bytes = p_u.astype("<f4").tobytes()
    bigram_bytes = p_b.astype("<f4").tobytes()
    trigram_bytes = p_t.astype("<f4").tobytes()
    lambda_bytes = np.array(lambdas, dtype="<f4").tobytes()
    out = header + mapping_bytes + unigram_bytes + bigram_bytes + trigram_bytes + lambda_bytes
    open(sys.argv[4], "wb").write(out)
    # Verify size
    expected = 20 + 256 + 4*(V + V*V + V*V*V + 3)
    assert len(out) == expected, f"Size mismatch: {len(out)} vs {expected}"
    print(f"Written {len(out)} bytes", file=sys.stderr)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int version = hd[1];
    int d = hd[2];
    int V = hd[3];
    int order = hd[4];
    if (version != 1 || d != 16 || V != 65 || order != 3) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    // Expected size: 20 + 256 + 4*(V + V*V + V*V*V + 3)
    long expected = 20 + 256 + 4L * (V + V*V + V*V*V + 3);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }
    // Pointers
    unsigned char *mapping = wb + 20;
    float *p_u = (float *)(mapping + 256);
    float *p_b = p_u + V;
    float *p_t = p_b + V * V;
    float *lambdas = p_t + V * V * V;
    float lambda_t = lambdas[0], lambda_b = lambdas[1], lambda_u = lambdas[2];
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Last two bytes of context
        unsigned char b1 = xi[d - 2];
        unsigned char b2 = xi[d - 1];
        int idx1 = mapping[b1];
        int idx2 = mapping[b2];
        float *probs;
        float combined[65];
        if (idx1 == 255 || idx2 == 255) {
            // Unknown byte: back off to unigram
            probs = p_u;
        } else {
            // Combine trigram, bigram, unigram
            float *pt = p_t + (idx1 * V + idx2) * V;
            float *pb = p_b + idx2 * V;
            for (int c = 0; c < V; c++) {
                combined[c] = lambda_t * pt[c] + lambda_b * pb[c] + lambda_u * p_u[c];
            }
            probs = combined;
        }
        unsigned char best = 0;
        for (int c = 1; c < V; c++) if (probs[c] > probs[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```