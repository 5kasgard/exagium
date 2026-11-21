```hypothesis
Family: label-mlp8 (single hidden layer of 8 ReLU units).
Why: The optdigits task has 64 ordinal pixel features with nonlinear class boundaries; a small MLP adds minimal capacity over linear regression while keeping inference fast. 8 hidden units should capture basic feature interactions and beat linear (96.1%) without the cost of larger MLPs (label-mlp16: 97.7% at 8.7k Ir/item).
Failure: If dev accuracy ≤ 96.1% or inference cost exceeds label-linear by >2×, the family does not improve the trade-off.
```

```python
"""Table family with a small hidden layer (H=8) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses only stdlib + numpy.
Hyperparameters (learning rate, L2) are chosen on the last 20% of train.in (tuning split),
then the model is retrained on all data. Exported in the same table format as label-linear."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8
# Tuning grid
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_stats(X):
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    return mu, sd


def encode(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def softmax_grad(logits, y):
    # logits: (n, C), y: (n,)
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_mlp(Z, y, H, lr, L2, epochs, batch, seed, Z_val=None, y_val=None):
    n, k = Z.shape
    C = int(y.max()) + 1
    rng = np.random.default_rng(seed)

    # Xavier initialization
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]

            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            # Backward
            g = softmax_grad(logits, yy)                 # (batch, C)
            gh = (g @ W2.T) * (a > 0)                    # (batch, H)

            grads = [
                z.T @ gh + L2 * W1,                      # dW1
                gh.sum(axis=0),                          # db1
                h.T @ g + L2 * W2,                       # dW2
                g.sum(axis=0)                            # db2
            ]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation check
        if Z_val is not None:
            a_val = Z_val @ W1 + b1
            h_val = np.maximum(a_val, 0)
            logits_val = h_val @ W2 + b2
            pred = logits_val.argmax(axis=1)
            val_acc = (pred == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]

    if best_params is not None:
        params = best_params
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    idx_train, idx_val = np.arange(split), np.arange(split, n)
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]

    # Grid search
    best_cfg = None
    best_val_acc = -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for i_lr, lr in enumerate(LR_GRID):
        for i_l2, L2 in enumerate(L2_GRID):
            seed = SEED + i_lr * len(L2_GRID) + i_l2
            mu, sd = compute_stats(X_train)
            Z_train = encode(X_train, mu, sd)
            Z_val = encode(X_val, mu, sd)
            params = train_mlp(Z_train, y_train, H, lr, L2, epochs, BATCH, seed, Z_val, y_val)
            # Evaluate on validation
            W1, b1, W2, b2 = params
            a_val = Z_val @ W1 + b1
            h_val = np.maximum(a_val, 0)
            logits_val = h_val @ W2 + b2
            val_acc = (logits_val.argmax(axis=1) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_cfg = (lr, L2, mu.copy(), sd.copy())
    print(f"Best tuning: lr={best_cfg[0]}, L2={best_cfg[1]}, val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    lr, L2, mu, sd = best_cfg
    Z_all = encode(X, mu, sd)
    epochs_all = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_mlp(Z_all, y, H, lr, L2, epochs_all, BATCH, SEED)
    W1, b1, W2, b2 = params

    # Build lookup tables T (d x 256 x H)
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        enc = ((np.arange(256, dtype=np.float32) - mu[j]) / sd[j])[:, None]  # (256,1)
        T[j] = enc * W1[j]  # broadcasting to (256, H)

    # Export
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


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