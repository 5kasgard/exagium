```hypothesis
Family: label-mlp8 (one-hidden-layer ReLU MLP with 8 hidden units, hyperparameters tuned on the last 20% of train.in).
Why: optdigits is a small non-linear problem; a tiny MLP adds minimal compute over linear (≈600 vs 640 multiply-adds/item) but captures feature interactions. Validation-tuned H=8 should beat linear accuracy (0.9613) at similar or lower cost.
Failure: if dev accuracy ≤ 0.9613 or instructions/item ≥ 5500.
```

```python
"""Table family with validation-tuned hidden size for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal features by z-scoring per feature on the training split.
Searches H in {4,8,12} and LR in {1e-3,3e-3,1e-2} on the last 20% of train.in (file order),
then retrains on all data with the best configuration.
Exports the same per-byte lookup table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid to evaluate on the validation split
GRID = [
    {"H": 4, "LR": 3e-3},
    {"H": 8, "LR": 3e-3},
    {"H": 12, "LR": 3e-3},
    {"H": 8, "LR": 1e-3},
    {"H": 8, "LR": 1e-2},
]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x width_j) for each feature."""
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


def encode(X, enc):
    """Apply encodings to X, return concatenated float32 matrix (n x total_width)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t. logits."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(X, y, C, H, LR, seed):
    """Train on (X,y) with given H, LR. Return (params, enc) where params = (W1,b1,W2,b2) or (W1,b1)."""
    rng = np.random.default_rng(seed)
    types = ["ordinal"] * X.shape[1]  # all features are ordinal per card
    enc = build_encoding(X, types)
    Z = encode(X, enc)
    n, k = Z.shape
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

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
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
                    g.sum(0),
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [
                    z.T @ g + L2 * W1,
                    g.sum(0),
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params, enc


def predict(params, enc, X):
    """Return predicted class indices for X."""
    Z = encode(X, enc)
    if len(params) == 4:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    return logits.argmax(axis=1).astype(np.int64)


def accuracy(params, enc, X, y):
    return (predict(params, enc, X) == y).mean()


def write_weights(card, enc, params, path):
    """Export weights in the table family format."""
    d = card["input"]["bytes"]
    C = card["classes"]
    H = params[2].shape[0] if len(params) == 4 else 0
    W = H if H else C
    # Build T: for each feature j, T_j = E_j @ W1[off:off+width_j]
    off = 0
    T_list = []
    for E in enc:
        width = E.shape[1]
        T_list.append((E.astype(np.float32) @ params[0][off:off + width]).astype(np.float32))
        off += width
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    open(path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(X_all)
    n_val = n // 5
    n_train = n - n_val
    # Split in file order: first 80% train, last 20% validation
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:], y_all[n_train:]

    best_acc = -1.0
    best_cfg = None
    best_params = None
    best_enc = None

    for cfg in GRID:
        H, LR = cfg["H"], cfg["LR"]
        # Use a deterministic seed derived from the config
        cfg_seed = SEED + H * 1000 + int(LR * 10000)
        params, enc = train_model(X_train, y_train, C, H, LR, cfg_seed)
        acc = accuracy(params, enc, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params
            best_enc = enc

    # Retrain on full data with best configuration
    H, LR = best_cfg["H"], best_cfg["LR"]
    final_seed = SEED + 9999  # distinct seed for final training
    final_params, final_enc = train_model(X_all, y_all, C, H, LR, final_seed)
    write_weights(card, final_enc, final_params, sys.argv[4])


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