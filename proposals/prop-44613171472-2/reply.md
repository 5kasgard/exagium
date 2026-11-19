```hypothesis
The label-hash-ngram family uses feature hashing to capture unigram, bigram, and trigram statistics in a fixed-size linear model. Character-level language modeling benefits from local n-gram interactions, which the linear model on independent bytes cannot capture. The hashing trick allows a compact model with fast inference (only additions). If it does not beat label-linear, it would indicate that n-gram interactions are not sufficient or hashing collisions degrade performance.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed n-gram configuration: unigrams (16), bigrams (15), trigrams (14) = 45 features per example
def hash_ngram(bytes_arr, pos, length, feature_size):
    # FNV-1a 32-bit
    h = 2166136261
    h ^= pos
    h = (h * 16777619) & 0xFFFFFFFF
    for k in range(length):
        h ^= bytes_arr[pos + k]
        h = (h * 16777619) & 0xFFFFFFFF
    return h & (feature_size - 1)

def compute_indices(X, feature_size):
    n = X.shape[0]
    indices = np.zeros((n, 45), dtype=np.int32)
    for i in range(n):
        idx = 0
        # unigrams
        for p in range(16):
            indices[i, idx] = hash_ngram(X[i], p, 1, feature_size)
            idx += 1
        # bigrams
        for p in range(15):
            indices[i, idx] = hash_ngram(X[i], p, 2, feature_size)
            idx += 1
        # trigrams
        for p in range(14):
            indices[i, idx] = hash_ngram(X[i], p, 3, feature_size)
            idx += 1
    return indices

def train_model(X, y, indices, C, feature_size, epochs, batch_size, lr, l2, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(feature_size, C)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            batch_indices = indices[idx]
            batch_y = y[idx]
            logits = np.zeros((len(idx), C), dtype=np.float32)
            for f in range(45):
                logits += W[batch_indices[:, f]]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(batch_y)), batch_y] -= 1.0
            grad = probs / len(batch_y)
            grad_W = np.zeros_like(W)
            flat_idx = batch_indices.ravel()
            grad_repeated = np.repeat(grad, 45, axis=0)
            np.add.at(grad_W, flat_idx, grad_repeated)
            grad_W += l2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad_W
            v = 0.999 * v + 0.001 * grad_W * grad_W
            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W

def evaluate(X, y, indices, W):
    n = X.shape[0]
    logits = np.zeros((n, W.shape[1]), dtype=np.float32)
    for f in range(45):
        logits += W[indices[:, f]]
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    split = n - n//5  # last 20% for validation
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    best_acc = -1
    best_W = None
    best_feature_size = 65536
    for feature_size in [32768, 65536, 131072]:
        print(f"Trying feature_size={feature_size}", file=sys.stderr)
        train_indices = compute_indices(X_train, feature_size)
        valid_indices = compute_indices(X_valid, feature_size)
        W = train_model(X_train, y_train, train_indices, C, feature_size, epochs=20, batch_size=64, lr=3e-3, l2=1e-4, seed=1)
        acc = evaluate(X_valid, y_valid, valid_indices, W)
        print(f"  Validation accuracy: {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_W = W
            best_feature_size = feature_size

    print(f"Best feature_size: {best_feature_size}, retraining on full data...", file=sys.stderr)
    full_indices = compute_indices(X, best_feature_size)
    W = train_model(X, y, full_indices, C, best_feature_size, epochs=30, batch_size=64, lr=3e-3, l2=1e-4, seed=1)

    magic = 0x484E474C  # "LNGH"
    out = struct.pack("<3i", magic, best_feature_size, C)
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

static uint32_t hash_ngram(const unsigned char *bytes, int pos, int length, uint32_t mask) {
    uint32_t h = 2166136261u;
    h ^= (uint32_t)pos;
    h *= 16777619u;
    for (int k = 0; k < length; k++) {
        h ^= bytes[pos + k];
        h *= 16777619u;
    }
    return h & mask;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    uint32_t magic = hd[0];
    if (magic != 0x484E474C) { fprintf(stderr, "bad magic\n"); return 3; }
    int feature_size = hd[1];
    int C = hd[2];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 12 + (long)feature_size * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    uint32_t mask = (uint32_t)feature_size - 1;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int p = 0; p < 16; p++) {
            uint32_t idx = hash_ngram(xi, p, 1, mask);
            const float *row = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        for (int p = 0; p < 15; p++) {
            uint32_t idx = hash_ngram(xi, p, 2, mask);
            const float *row = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        for (int p = 0; p < 14; p++) {
            uint32_t idx = hash_ngram(xi, p, 3, mask);
            const float *row = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(scores);
    return 0;
}
```