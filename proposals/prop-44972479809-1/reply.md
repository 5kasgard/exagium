```hypothesis
Family: label-mlp-tune (one-hidden-layer MLP with hyperparameter search on the tuning split).
Reason: optdigits has non-linear pixel interactions; a small MLP can capture them. Tuning hidden size (8,16,32,64), L2 (1e-4,1e-3,1e-2) and LR (1e-3,3e-3,1e-2) on the last 20% of train.in should find a configuration that beats label-linear (0.9613) and approaches label-mlp16 (0.9770) with similar inference cost.
Failure signal: dev accuracy ≤ 0.9613 or inference cost (Ir/item) more than 2× label-linear.
```

```python
import json, math, struct, sys, itertools

import numpy as np

# Fixed training hyperparameters
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning
HIDDEN_SIZES = [8, 16, 32, 64]
L2_VALS = [1e-4, 1e-3, 1e-2]
LR_VALS = [1e-3, 3e-3, 1e-2]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit=True, stats=None):
    """Return (Z, new_stats) where Z is float32 encoded matrix, stats is list per feature."""
    n, d = X.shape
    if fit:
        stats = []
    encoders = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            if fit:
                mu, sd = v.mean(), v.std() or 1.0
                stats.append((mu, sd))
            else:
                mu, sd = stats[j]
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            encoders.append(E)
        else:
            if fit:
                vals = np.unique(X[:, j])
                stats.append(vals)
            else:
                vals = stats[j]
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            encoders.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(encoders)], axis=1).astype(np.float32)
    return Z, stats


def train_mlp(Z, y, H, C, LR, L2, rng, max_epochs):
    n, k = Z.shape
    W = H
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    W2 = (rng.uniform(-1, 1, (W, C)) * math.sqrt(6.0 / W)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            gh = (g @ W2.T) * (a > 0)
            grads = [
                z.T @ gh + L2 * W1,
                gh.sum(0),
                h.T @ g + L2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n_all = len(X_all)
    # tuning split: last 20% in file order
    n_tune = n_all // 5
    n_train = n_all - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    # Encode training split once (stats from train split)
    Z_train, stats = encode_features(X_train, types, fit=True)
    Z_tune, _ = encode_features(X_tune, types, fit=False, stats=stats)

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for H, L2, LR in itertools.product(HIDDEN_SIZES, L2_VALS, LR_VALS):
        # deterministic seed per config
        cfg_seed = hash((H, L2, LR)) & 0xFFFFFFFF
        rng = np.random.default_rng(cfg_seed)
        max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        params = train_mlp(Z_train, y_train, H, C, LR, L2, rng, max_epochs)
        acc = evaluate(Z_tune, y_tune, params)
        if acc > best_acc or (acc == best_acc and (H < best_cfg[0] or (H == best_cfg[0] and L2 < best_cfg[1]) or (H == best_cfg[0] and L2 == best_cfg[1] and LR < best_cfg[2]))):
            best_acc = acc
            best_cfg = (H, L2, LR)
            best_params = params

    # Retrain on full data with best config
    H, L2, LR = best_cfg
    Z_full, stats_full = encode_features(X_all, types, fit=True)
    rng = np.random.default_rng(SEED)
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    final_params = train_mlp(Z_full, y_all, H, C, LR, L2, rng, max_epochs)
    W1, b1, W2, b2 = final_params

    # Build per-byte lookup tables for first layer (same format as label-linear)
    d = X_all.shape[1]
    W = H
    off = 0
    T_list = []
    for j, t in enumerate(types):
        if t == "ordinal":
            mu, sd = stats_full[j]
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            dim = 1
        else:
            vals = stats_full[j]
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            dim = len(vals)
        T_list.append((E @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # (d, 256, W)

    # Pack weights: magic, d, H, C; then T (d*256*W float32), b1 (W), W2 (H*C), b2 (C)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
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