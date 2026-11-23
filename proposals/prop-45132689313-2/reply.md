```hypothesis
We propose a hashed n-gram linear model (family label-hashed-ngram) that combines unigram and bigram features via feature hashing into a fixed-size weight matrix. This captures local character interactions (bigrams) which are crucial for language modeling, while remaining extremely fast at inference (only 31 feature lookups per example). It should beat the linear baseline (which only uses unigrams) on accuracy, with similar CPU cost. If it does not beat linear on dev accuracy, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS = 20
L2 = 1e-4
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def extract_features(context_bytes, bins):
    """context_bytes: bytes of length 16. Returns list of 31 feature indices."""
    feats = []
    for p in range(16):
        b = context_bytes[p]
        feats.append((p * 257 + b) % bins)
    for p in range(15):
        b1 = context_bytes[p]
        b2 = context_bytes[p+1]
        feats.append((p * 66049 + b1 * 257 + b2) % bins)
    return feats

def train_model(X_feats, y, bins, C, lr, l2, batch, max_epochs, seed):
    n = len(y)
    rng = np.random.default_rng(seed)
    W = (rng.uniform(-1, 1, (bins, C)) * math.sqrt(6.0 / bins)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            batch_feats = X_feats[idx]
            batch_y = y[idx]
            logits = np.sum(W[batch_feats], axis=1) + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(batch_y)), batch_y] -= 1.0
            grad_logits = probs / len(batch_y)
            grad_b = grad_logits.sum(axis=0)
            grad_W = np.zeros_like(W)
            flat_feats = batch_feats.ravel()
            repeated_grad = np.repeat(grad_logits, 31, axis=0)
            np.add.at(grad_W, flat_feats, repeated_grad)
            grad_W += l2 * W
            grad_b += l2 * b
            t += 1
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b

def evaluate(W, b, X_feats, y):
    logits = np.sum(W[X_feats], axis=1) + b
    preds = np.argmax(logits, axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]
    C = card["classes"]
    X_recs = records(train_in)
    y_recs = records(train_key)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    best_acc = -1
    best_params = None
    best_W = None
    best_b = None
    for bins in [4096, 8192, 16384]:
        for lr in [1e-3, 3e-3]:
            train_feats = np.array([extract_features(x.tobytes(), bins) for x in X_train], dtype=np.int32)
            tune_feats = np.array([extract_features(x.tobytes(), bins) for x in X_tune], dtype=np.int32)
            W, b = train_model(train_feats, y_train, bins, C, lr, L2, BATCH, MAX_EPOCHS, SEED)
            acc = evaluate(W, b, tune_feats, y_tune)
            print(f"bins={bins}, lr={lr}, tune_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_params = (bins, lr)
                best_W, best_b = W, b
    print(f"Best: bins={best_params[0]}, lr={best_params[1]}, acc={best_acc:.4f}")
    bins, lr = best_params
    all_feats = np.array([extract_features(x.tobytes(), bins) for x in X], dtype=np.int32)
    W, b = train_model(all_feats, y, bins, C, lr, L2, BATCH, MAX_EPOCHS, SEED)
    magic = 0x484E4731  # 'HNG1'
    out = struct.pack('<3i', magic, bins, C)
    out += W.astype('<f4').tobytes()
    out += b.astype('<f4').tobytes()
    open(weights_out, "wb").write(out)

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

static inline uint32_t hash_unigram(int pos, uint8_t byte, uint32_t bins) {
    return (pos * 257 + byte) % bins;
}
static inline uint32_t hash_bigram(int pos, uint8_t byte1, uint8_t byte2, uint32_t bins) {
    return (pos * 66049 + byte1 * 257 + byte2) % bins;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int bins = hd[1], C = hd[2];
    if (C < 1 || C > 256 || bins < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 12 + (long)bins * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)bins * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b[c];
        for (int p = 0; p < 16; p++) {
            uint32_t h = hash_unigram(p, xi[p], bins);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }
        for (int p = 0; p < 15; p++) {
            uint32_t h = hash_bigram(p, xi[p], xi[p+1], bins);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```