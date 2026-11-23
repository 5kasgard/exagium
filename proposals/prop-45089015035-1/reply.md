```hypothesis
Propose label-mlp32: a one-hidden-layer MLP with 32 ReLU units, trained with Adam on per-pixel z-scored features.
The small optdigits dataset (64 ordinal features, 2867 samples) benefits from mild nonlinearity without overfitting.
Hyperparameters (H, LR, L2) are selected on the last 20% of train.in (file order), then retrained on all data.
If dev accuracy does not exceed label-linear (0.9613) by at least 0.5%, the extra compute is not justified.
```

```python
"""label-mlp32: one-hidden-layer MLP with 32 ReLU units for optdigits.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each pixel (0..16) as z-score, builds per-byte lookup tables for fast inference.
Hyperparameters tuned on last 20% of train.in (file order), then retrained on full train set.
Weight format identical to table family (magic 0x54424C31, d, H, C, T, b1, W2, b2)."""
import json, math, struct, sys

import numpy as np

# Fixed seed for top-level determinism
SEED = 1
# Hyperparameter grid to search on validation split
GRID = [
    {"H": 16, "LR": 3e-3, "L2": 1e-4, "BATCH": 64},
    {"H": 32, "LR": 3e-3, "L2": 1e-4, "BATCH": 64},
    {"H": 64, "LR": 3e-3, "L2": 1e-4, "BATCH": 64},
    {"H": 32, "LR": 1e-3, "L2": 1e-4, "BATCH": 64},
    {"H": 32, "LR": 1e-2, "L2": 1e-4, "BATCH": 64},
    {"H": 32, "LR": 3e-3, "L2": 1e-3, "BATCH": 64},
]
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encodings(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode_batch(X, enc):
    """Transform X (n x d) using encodings to Z (n x total_dim)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def accuracy(Z, y, W1, b1, W2, b2, H, C):
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def train_model(X, y, config, rng):
    """Train on given X,y with config, return (W1, b1, W2, b2, enc)."""
    H = config["H"]
    LR = config["LR"]
    L2 = config["L2"]
    BATCH = config["BATCH"]
    n, d = X.shape
    C = 10  # from card

    # Encodings on this training set
    enc = compute_encodings(X, ["ordinal"] * d)
    Z = encode_batch(X, enc)
    k = Z.shape[1]
    W = H if H else C

    # Initialize parameters
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
        W2 = b2 = None

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    if H:
        return W1, b1, W2, b2, enc
    else:
        return W1, b1, None, None, enc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]

    # Split: last 20% for validation (file order)
    n_val = max(1, int(0.2 * n))
    train_idx = np.arange(n - n_val)
    val_idx = np.arange(n - n_val, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Grid search
    best_acc = -1.0
    best_config = None
    base_rng = np.random.default_rng(SEED)
    for i, config in enumerate(GRID):
        # Derive a deterministic seed for this config
        config_seed = SEED + i + 1
        rng = np.random.default_rng(config_seed)
        W1, b1, W2, b2, enc = train_model(X_train, y_train, config, rng)
        Z_val = encode_batch(X_val, enc)
        acc = accuracy(Z_val, y_val, W1, b1, W2, b2, config["H"], C)
        if acc > best_acc:
            best_acc = acc
            best_config = config

    # Retrain on full data with best config
    print(f"Selected config: {best_config}, val acc: {best_acc:.4f}", file=sys.stderr)
    final_rng = np.random.default_rng(SEED + 999)
    W1, b1, W2, b2, enc = train_model(X, y, best_config, final_rng)

    # Build lookup tables T for inference
    H = best_config["H"]
    W = H if H else C
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    # Pack weights: magic, d, H, C; T (d x 256 x W), b1 (W), [W2 (H x C), b2 (C)]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
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