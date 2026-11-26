```hypothesis
We propose "label-wide-mlp": a two-layer MLP with per-position embeddings and a wide hidden layer (up to 256 units). 
The linear model (label-linear) sums position-specific weights; adding a ReLU layer allows pairwise interactions. 
We tune hidden size, learning rate, and L2 on the last 20% of train.in, then retrain on all data. 
Expected: dev accuracy > 0.34 (beating label-linear) with inference cost < 15k Ir/item. 
Failure: accuracy ≤ label-linear or cost ≥ label-gbdt.
```

```python
"""label-wide-mlp: two-layer MLP with per-position embeddings, tuned on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Weights format: magic 0x5749444D, int32 d, H, C; float32 T[d][256][W], b1[W]; if H>0: float32 W2[H][C], b2[C] (W=H)."""
import json, math, struct, sys

import numpy as np

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64

# Grid for tuning split (last 20% of train.in)
CONFIGS = [
    {"H": 0,   "LR": 3e-3, "L2": 1e-4},
    {"H": 32,  "LR": 3e-3, "L2": 1e-4},
    {"H": 64,  "LR": 2e-3, "L2": 1e-4},
    {"H": 128, "LR": 1e-3, "L2": 1e-4},
    {"H": 256, "LR": 1e-3, "L2": 1e-4},
]

MAGIC = 0x5749444D  # "WIDM"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc, dims = [], []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
            dims.append(1)
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            dims.append(len(vals))
    return enc, dims

def encode_batch(X, enc, dims, offsets):
    """X: (n, d) uint8 -> Z: (n, total_dim) float32"""
    n = X.shape[0]
    total = sum(dims)
    Z = np.empty((n, total), np.float32)
    for j, (E, off, dim) in enumerate(zip(enc, offsets, dims)):
        Z[:, off:off+dim] = E[X[:, j]]
    return Z

def train_model(X, y, enc, dims, offsets, C, config, epochs_override=None):
    """Train on given data, return weights dict."""
    H, LR, L2 = config["H"], config["LR"], config["L2"]
    W = H if H else C
    total_dim = sum(dims)
    n = X.shape[0]
    Z = encode_batch(X, enc, dims, offsets)
    rng = np.random.default_rng(SEED)
    # init
    if H:
        W1 = (rng.uniform(-1, 1, (total_dim, H)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((total_dim, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = epochs_override if epochs_override is not None else min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    # build per-position lookup tables T
    off = 0
    T = []
    for j, (E, dim) in enumerate(zip(enc, dims)):
        T.append((E @ params[0][off:off+dim]).astype(np.float32))
        off += dim
    weights = {"T": T, "b1": params[1].astype(np.float32)}
    if H:
        weights["W2"] = params[2].astype(np.float32)
        weights["b2"] = params[3].astype(np.float32)
    return weights

def predict(X, enc, dims, offsets, weights, C):
    H = weights["W2"].shape[0] if "W2" in weights else 0
    W = H if H else C
    Z = encode_batch(X, enc, dims, offsets)
    if H:
        a = Z @ weights["T_full"]?  # We'll compute via T lookup for speed
    # For simplicity, reuse the same forward logic as training but without batches.
    # Since n is small (tuning set), we can just do the matrix multiplies.
    if H:
        a = Z @ np.vstack(weights["T"]) + weights["b1"]
        h = np.maximum(a, 0)
        scores = h @ weights["W2"] + weights["b2"]
    else:
        scores = Z @ np.vstack(weights["T"]) + weights["b1"]
    return scores.argmax(axis=1)

def accuracy(X, y, enc, dims, offsets, weights, C):
    pred = predict(X, enc, dims, offsets, weights, C)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    # tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_tune, y_tune = X_all[tune_idx], y_all[tune_idx]

    # Build encoding on training split (for tuning)
    enc_train, dims_train = build_encoding(X_train, types)
    offsets = np.cumsum([0] + dims_train[:-1])

    best_acc, best_cfg, best_weights = -1, None, None
    for cfg in CONFIGS:
        w = train_model(X_train, y_train, enc_train, dims_train, offsets, C, cfg, epochs_override=20)
        # For prediction, we need to stack T into a full matrix? We'll compute accuracy using the same forward.
        # We'll reuse the training forward by constructing Z_tune.
        Z_tune = encode_batch(X_tune, enc_train, dims_train, offsets)
        H = cfg["H"]
        if H:
            a = Z_tune @ np.vstack(w["T"]) + w["b1"]
            h = np.maximum(a, 0)
            scores = h @ w["W2"] + w["b2"]
        else:
            scores = Z_tune @ np.vstack(w["T"]) + w["b1"]
        acc = (scores.argmax(axis=1) == y_tune).mean()
        if acc > best_acc:
            best_acc, best_cfg, best_weights = acc, cfg, w
        print(f"Config {cfg}: tune acc = {acc:.4f}", file=sys.stderr)

    print(f"Best config: {best_cfg} with acc {best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    enc_all, dims_all = build_encoding(X_all, types)
    offsets_all = np.cumsum([0] + dims_all[:-1])
    final_weights = train_model(X_all, y_all, enc_all, dims_all, offsets_all, C, best_cfg)

    # Write weights.bin
    H = best_cfg["H"]
    W = H if H else C
    # Flatten T: list of (256, W) for each position
    T_flat = np.stack(final_weights["T"]).astype("<f4")  # (d, 256, W)
    b1 = final_weights["b1"].astype("<f4")
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += T_flat.tobytes()
    out += b1.tobytes()
    if H:
        out += final_weights["W2"].astype("<f4").tobytes()
        out += final_weights["b2"].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-wide-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index.
 * Weights format: magic 0x5749444D, int32 d, H, C; float32 T[d][256][W], b1[W]; if H>0: float32 W2[H][C], b2[C] (W=H). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x5749444D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        if (H) {
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) sc[c] += h * W2[(long)u * C + c];
            }
        } else {
            for (int c = 0; c < C; c++) sc[c] = acc[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```