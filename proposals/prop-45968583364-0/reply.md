```hypothesis
Family: label-embed-mlp (one-hidden-layer MLP with per-feature embeddings for all 14 columns).
Why: Treats ordinal features as categorical (256 embeddings each) to capture non-monotonic effects and learns feature interactions via an 8-unit ReLU layer. Trained with Adam on the tuning split (last 20%) for early stopping, then retrained on all data. Expects >0.8524 dev accuracy (beating label-linear) with inference cost ~1000 Ir/item.
Failure: If dev accuracy ≤ 0.8524 or Ir/item > 2000, the hypothesis is not supported.
```

```python
"""label-embed-mlp: one-hidden-layer MLP with per-feature embeddings (size 8) for all columns.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib+numpy only.
Exports weights in the same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8          # hidden units (embedding size W = H)
C = 2          # classes
SEED = 42      # global seed
BATCH = 64
MAX_EPOCHS_GRID = 10   # epochs for hyperparameter search
MAX_EPOCHS_FULL = 50   # epochs for final training on all data
LR_CANDIDATES = [1e-3, 3e-3]
L2_CANDIDATES = [1e-4, 1e-5]


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, T, b1, W2, b2, H):
    n = X.shape[0]
    acc = np.tile(b1, (n, 1))
    for j in range(X.shape[1]):
        acc += T[j, X[:, j], :]
    if H:
        h = np.maximum(acc, 0)
        logits = h @ W2 + b2
    else:
        logits = acc
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(X_tr, y_tr, X_val, y_val, H, lr, L2, epochs, batch_size, seed):
    n, d = X_tr.shape
    W = H
    rng = np.random.default_rng(seed)
    # Initialize parameters
    T = (rng.normal(0, 0.01, (d, 256, W))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    W2 = (rng.normal(0, 0.01, (H, C)) * np.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [T, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_val = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]
            bs = xb.shape[0]
            # Forward
            acc = b1 + np.zeros((bs, W), np.float32)
            for j in range(d):
                acc += T[j, xb[:, j], :]
            if H:
                h = np.maximum(acc, 0)
                logits = h @ W2 + b2
            else:
                logits = acc
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs
            # Backward
            if H:
                grad_h = grad_logits @ W2.T
                grad_h *= (acc > 0)
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(axis=0)
                grad_acc = grad_h
            else:
                grad_W2 = None
                grad_b2 = None
                grad_acc = grad_logits
            grad_b1 = grad_acc.sum(axis=0)
            grad_T = np.zeros_like(T)
            for j in range(d):
                np.add.at(grad_T[j], xb[:, j], grad_acc)
            grad_T += L2 * T
            # Adam update
            grads = [grad_T, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if gr is None:
                    continue
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation
        val_acc = evaluate(X_val, y_val, T, b1, W2, b2, H)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val


def write_weights(path, d, H, C, T, b1, W2, b2):
    W = H if H else C
    magic = 0x54424C31
    header = struct.pack("<4i", magic, d, H, C)
    body = (T.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            (W2.astype("<f4").tobytes() if H else b"") +
            (b2.astype("<f4").tobytes() if H else b""))
    with open(path, "wb") as f:
        f.write(header + body)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_val = -1.0
    best_hparams = None
    best_params = None
    for lr in LR_CANDIDATES:
        for L2 in L2_CANDIDATES:
            seed = SEED + hash((lr, L2)) % 2**32
            params, val_acc = train_model(X_tr, y_tr, X_val, y_val,
                                          H, lr, L2, MAX_EPOCHS_GRID, BATCH, seed)
            if val_acc > best_val:
                best_val = val_acc
                best_hparams = (lr, L2)
                best_params = params
    print(f"Best tuning: lr={best_hparams[0]}, L2={best_hparams[1]}, val_acc={best_val:.4f}")

    # Retrain on full data with best hyperparameters
    lr, L2 = best_hparams
    full_params, _ = train_model(X, y, X_val, y_val,  # validation not used but kept for interface
                                 H, lr, L2, MAX_EPOCHS_FULL, BATCH, SEED)
    T, b1, W2, b2 = full_params
    write_weights(sys.argv[4], d, H, C, T, b1, W2, b2)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```