```hypothesis
Family: label-hash (hashed linear model with unigram+bigram features).
Why: Captures local character interactions (bigrams) via feature hashing, adding minimal overhead vs label-linear. Hashing reduces parameters vs full one-hot bigrams, enabling better generalization on small data. Expect to beat label-linear (0.3375) by modeling pairwise context while keeping inference fast (~2x cost). Failure if validation accuracy not >0.34.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
B_VALUES = [16384, 32768, 65536]
LR_VALUES = [0.01, 0.003, 0.001]
L2_VALUES = [1e-4, 1e-5]
SEARCH_EPOCHS = 5
FINAL_EPOCHS = 20
BATCH = 64
SEED = 1

MAGIC = 0x4841534C  # "HASL" little-endian
FEATURE_TYPE = 0    # unigram + bigram


def fnv1a_hash(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def hash_unigram(pos: int, byte: int, B: int) -> int:
    return fnv1a_hash(bytes([0, pos, byte])) % B


def hash_bigram(pos: int, b1: int, b2: int, B: int) -> int:
    return fnv1a_hash(bytes([1, pos, b1, b2])) % B


def extract_features(X: np.ndarray, B: int) -> np.ndarray:
    n, d = X.shape
    assert d == 16
    feats = np.zeros((n, 31), dtype=np.int32)
    for i in range(n):
        idx = 0
        for p in range(16):
            feats[i, idx] = hash_unigram(p, X[i, p], B)
            idx += 1
        for p in range(15):
            feats[i, idx] = hash_bigram(p, X[i, p], X[i, p + 1], B)
            idx += 1
    return feats


def train_model(X, y, feats, B, C, lr, l2, epochs, batch, rng):
    n = X.shape[0]
    W = (rng.uniform(-1, 1, (B, C)) * math.sqrt(6.0 / B)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            f = feats[idx]
            yy = y[idx]
            scores = b + W[f].sum(axis=1)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = probs / len(yy)
            dW = np.zeros_like(W)
            np.add.at(dW, f.ravel(), np.repeat(grad_scores, 31, axis=0))
            dW += l2 * W
            db = grad_scores.sum(axis=0)
            grads = [dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def accuracy(X, y, feats, W, b):
    scores = b + W[feats].sum(axis=1)
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_acc = -1.0
    best_params = None
    best_W = best_b = None
    rng = np.random.default_rng(SEED)

    for B in B_VALUES:
        feats_tr = extract_features(X_tr, B)
        feats_va = extract_features(X_va, B)
        for lr in LR_VALUES:
            for l2 in L2_VALUES:
                print(f"Trying B={B}, lr={lr}, l2={l2}", file=sys.stderr)
                W, b = train_model(X_tr, y_tr, feats_tr, B, C, lr, l2, SEARCH_EPOCHS, BATCH, rng)
                acc = accuracy(X_va, y_va, feats_va, W, b)
                print(f"  val acc={acc:.4f}", file=sys.stderr)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (B, lr, l2)
                    best_W, best_b = W.copy(), b.copy()

    print(f"Best: B={best_params[0]}, lr={best_params[1]}, l2={best_params[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    B, lr, l2 = best_params
    feats_full = extract_features(X, B)
    W_final, b_final = train_model(X, y, feats_full, B, C, lr, l2, FINAL_EPOCHS, BATCH, rng)

    header = struct.pack("<5i", MAGIC, B, C, d, FEATURE_TYPE)
    weight_bytes = W_final.astype("<f4").tobytes()
    bias_bytes = b_final.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(weight_bytes)
        f.write(bias_bytes)


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

static uint32_t fnv1a_hash(const unsigned char *data, size_t len) {
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

static uint32_t hash_unigram(int pos, int byte, uint32_t B) {
    unsigned char buf[3] = {0, (unsigned char)pos, (unsigned char)byte};
    return fnv1a_hash(buf, 3) % B;
}

static uint32_t hash_bigram(int pos, int byte1, int byte2, uint32_t B) {
    unsigned char buf[4] = {1, (unsigned char)pos, (unsigned char)byte1, (unsigned char)byte2};
    return fnv1a_hash(buf, 4) % B;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    uint32_t magic = hd[0];
    if (magic != 0x4841534C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t B = hd[1];
    uint32_t C = hd[2];
    uint32_t d = hd[3];
    uint32_t feature_type = hd[4];
    if (feature_type != 0) { fprintf(stderr, "unsupported feature type\n"); return 3; }
    if (d != 16) { fprintf(stderr, "unexpected input bytes %u\n", d); return 3; }
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes %u\n", C); return 3; }
    long expected = 20 + (long)B * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    float *weights = (float *)(wb + 20);
    float *bias = weights + (long)B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        for (uint32_t c = 0; c < C; c++) scores[c] = bias[c];
        for (int p = 0; p < 16; p++) {
            uint32_t bin = hash_unigram(p, xi[p], B);
            float *w = weights + (long)bin * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int p = 0; p < 15; p++) {
            uint32_t bin = hash_bigram(p, xi[p], xi[p + 1], B);
            float *w = weights + (long)bin * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```