```hypothesis
A one-hidden-layer MLP (64 ReLU units) on position-specific one-hot encodings captures pairwise byte interactions that a linear model misses. The 16-byte context from TinyShakespeare has strong local n-gram structure; a hidden layer of 64 adds ~266k multiply-adds (similar to linear's 266k) but with non-linearity. Hyperparameters (hidden size, learning rate) are chosen on the last 20% of train.in, then retrained on all data. If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%, the hypothesis fails.
```

```python
"""MLP family for label tasks with fixed-size byte inputs: one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Chooses hidden size and learning rate on the last 20% of train.in (tuning split),
then trains on the full training set and writes weights in the label-linear format.
"""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS_TUNE = 30   # for hyperparameter search
MAX_EPOCHS_FULL = 100  # for final training

# Hyperparameter grid to search on the tuning split
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3, 1e-2]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoder(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) for each position."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    return enc


def encode_batch(X, enc):
    """X: (n, d) uint8 -> Z: (n, total_dim) float32"""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - one_hot) / n"""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    P = exp / exp.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return P / len(y)


def train_mlp(Z_train, y_train, Z_val, y_val, H, C, lr, epochs, rng):
    """Train MLP on train set, return (params, val_acc) where params = [W1,b1,W2,b2]"""
    n, k = Z_train.shape
    # Xavier init
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0, dtype=np.float32)
            logits = h @ W2 + b2
            # gradient
            g = softmax_grad(logits, yy)          # (batch, C)
            gh = (g @ W2.T) * (a > 0)             # (batch, H)
            grads = [
                z.T @ gh + L2 * W1,               # W1
                gh.sum(0),                        # b1
                h.T @ g + L2 * W2,                # W2
                g.sum(0)                          # b2
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        a_val = Z_val @ W1 + b1
        h_val = np.maximum(a_val, 0, dtype=np.float32)
        logits_val = h_val @ W2 + b2
        preds = logits_val.argmax(axis=1)
        val_acc = (preds == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    return best_params, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    types = card["input"]["types"]

    # Split: last 20% for tuning (by file order)
    n = X.shape[0]
    n_train = int(0.8 * n)
    idx_train = np.arange(n_train)
    idx_val = np.arange(n_train, n)

    # Build encoder on FULL training data (input only, no label leakage)
    enc = build_encoder(X, types)
    Z = encode_batch(X, enc)
    k = Z.shape[1]

    rng = np.random.default_rng(SEED)

    # Hyperparameter search
    best_score = -1.0
    best_H = H_CHOICES[0]
    best_lr = LR_CHOICES[0]
    best_params = None

    for H in H_CHOICES:
        for lr in LR_CHOICES:
            params, val_acc = train_mlp(Z[idx_train], y[idx_train], Z[idx_val], y[idx_val],
                                        H, C, lr, MAX_EPOCHS_TUNE, rng)
            if val_acc > best_score:
                best_score = val_acc
                best_H = H
                best_lr = lr
                best_params = params

    # Retrain on full data with best hyperparameters
    final_params, _ = train_mlp(Z, y, Z, y, best_H, C, best_lr, MAX_EPOCHS_FULL, rng)
    W1, b1, W2, b2 = final_params

    # Build per-position lookup tables T[j] = E_j @ W1[offset:offset+dim_j]
    T_list = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T_list.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim

    # Pack weights: magic, d, H, C; then T (d * 256 * W), b1 (W), W2 (H*C), b2 (C)
    W = best_H
    magic = 0x54424C31  # "TBL1"
    header = struct.pack("<4i", magic, d, best_H, C)
    T_bytes = np.stack(T_list).astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + T_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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