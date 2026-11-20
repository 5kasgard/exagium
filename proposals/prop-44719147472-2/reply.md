```hypothesis
The label-hashngram family uses hashed n-gram features (lengths 1–4) with a linear classifier.
It captures local character patterns better than per-position one-hot (label-linear) while keeping
inference fast via a fixed-size weight matrix. It should beat label-linear's 0.3375 dev accuracy.
If it fails to exceed 0.34, the approach is not effective for this task.
```

```python
"""Hash n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split in train.py)
SEED = 1
MAX_NGRAM = 4
M_CANDIDATES = [4096, 8192]
LR_CANDIDATES = [0.01, 0.03]
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
SAMPLES_PER_EPOCH = 600000

MAGIC = 0x484E4748  # 'HNGH'


def fnv1a(data, seed=0x811c9dc5):
    h = seed
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_features(X, M, max_ngram):
    """X: (n, d) uint8. Returns (n, M) float32 dense feature matrix."""
    n, d = X.shape
    feat = np.zeros((n, M), dtype=np.float32)
    for i in range(n):
        row = X[i]
        for L in range(1, max_ngram + 1):
            for start in range(d - L + 1):
                ngram = bytes(row[start:start + L])
                h = fnv1a(ngram) % M
                feat[i, h] += 1.0
    return feat


def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    split = int(n * (1 - val_frac))
    return X[:split], y[:split], X[split:], y[split:]


def softmax_grad(scores, y):
    # scores: (batch, C), y: (batch,) int
    batch = scores.shape[0]
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch


def evaluate(W, b, X_val, y_val, M, max_ngram, batch_size=1024):
    n = X_val.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        Xb = X_val[i:i + batch_size]
        yb = y_val[i:i + batch_size]
        feat = compute_features(Xb, M, max_ngram)
        scores = feat @ W + b
        pred = scores.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X_train, y_train, X_val, y_val, M, lr, l2, epochs, rng):
    C = 65
    W = np.zeros((M, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = X_train.shape[0]
    steps_per_epoch = max(1, math.ceil(SAMPLES_PER_EPOCH / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for step in range(steps_per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            if len(idx) == 0:
                break
            Xb = X_train[idx]
            yb = y_train[idx]
            feat = compute_features(Xb, M, MAX_NGRAM)
            scores = feat @ W + b
            grad = softmax_grad(scores, yb)  # (batch, C)
            gW = feat.T @ grad + l2 * W
            gb = grad.sum(axis=0) + l2 * b
            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb * gb
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    val_acc = evaluate(W, b, X_val, y_val, M, MAX_NGRAM)
    return W, b, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert C == 65

    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning (validation)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, 0.2)

    # Grid search on tuning split
    best_acc = -1
    best_params = None
    best_W, best_b = None, None
    for M in M_CANDIDATES:
        for lr in LR_CANDIDATES:
            W, b, acc = train_model(X_tr, y_tr, X_val, y_val, M, lr, L2, TUNE_EPOCHS, rng)
            if acc > best_acc:
                best_acc = acc
                best_params = (M, lr)
                best_W, best_b = W, b
    print(f"Tuning best: M={best_params[0]}, lr={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best params
    M_final, lr_final = best_params
    W_final, b_final, _ = train_model(X, y, X_val, y_val, M_final, lr_final, L2, FINAL_EPOCHS, rng)

    # Save weights
    # header: magic, M, C, max_ngram (all int32)
    header = struct.pack("<4i", MAGIC, M_final, C, MAX_NGRAM)
    # bias: float32[C]
    bias_bytes = b_final.astype("<f4").tobytes()
    # weight matrix: float32[M*C] row-major
    weight_bytes = W_final.astype("<f4").tobytes()
    out = header + bias_bytes + weight_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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

static uint32_t fnv1a(const unsigned char *data, uint32_t len) {
    uint32_t h = 0x811c9dc5;
    for (uint32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4748) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t M = hd[1], C = hd[2], max_ngram = hd[3];
    if (C < 1 || C > 256 || max_ngram != 4) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * C + 4L * M * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (uint32_t c = 0; c < C; c++) scores[c] = bias[c];

        for (uint32_t L = 1; L <= max_ngram; L++) {
            for (uint32_t start = 0; start + L <= 16; start++) {
                uint32_t h = fnv1a(xi + start, L) % M;
                const float *wrow = W + (long)h * C;
                for (uint32_t c = 0; c < C; c++) scores[c] += wrow[c];
            }
        }

        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```