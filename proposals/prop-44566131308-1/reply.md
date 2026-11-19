```hypothesis
Family: label-mlp8 – a single-hidden-layer MLP with 8 ReLU units, trained with Adam on z-scored ordinal features.
Why: The 64 ordinal pixel features of optdigits likely have non-linear interactions that a linear model misses; 8 hidden units add minimal compute (≈592 multiply-adds vs linear’s 640) while providing non-linearity. Validation-tuned learning rate and L2 prevent overfitting on the small dataset.
Failure: If dev accuracy ≤ 0.9613 (label-linear’s) or instructions per item exceed label-linear’s 5058.
```

```python
"""label-mlp8: Table family with one ReLU layer (H=8) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses LR and L2 on last 20% of train.in, then trains on all data.
Exported format: magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W], W2[H][C], b2[C] (W=H)."""
import json, math, struct, sys

import numpy as np

H = 8
SEED = 1
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-5, 1e-4, 1e-3]
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


def encode_features(X, types, fit=True, stats=None):
    """Return (encoders, Z) where encoders is list of arrays (256, dim_j) and Z is (n, k)."""
    n, d = X.shape
    encoders = []
    if fit:
        stats = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            if fit:
                mu, sd = v.mean(), v.std() or 1.0
                stats.append((mu, sd))
            else:
                mu, sd = stats[j]
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256,1)
        else:
            if fit:
                vals = np.unique(X[:, j])
                stats.append(vals)
            else:
                vals = stats[j]
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        encoders.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(encoders)], axis=1).astype(np.float32)
    return encoders, Z, stats


def train_mlp(Z, y, C, H, lr, l2, rng, epochs, batch):
    n, k = Z.shape
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            gh = (g @ W2.T) * (a > 0)
            grads = [
                z.T @ gh + l2 * W1,
                gh.sum(0),
                h.T @ g + l2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def accuracy(Z, y, params, C, H):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    val_start = int(0.8 * n)
    # Split for tuning
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]

    # Encode training part to get stats
    encoders_train, Z_train, stats = encode_features(X_train, types, fit=True)
    _, Z_val, _ = encode_features(X_val, types, fit=False, stats=stats)

    best_acc = -1.0
    best_lr, best_l2 = LR_GRID[0], L2_GRID[0]
    for lr in LR_GRID:
        for l2 in L2_GRID:
            rng = np.random.default_rng(SEED)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
            params = train_mlp(Z_train, y_train, C, H, lr, l2, rng, epochs, BATCH)
            acc = accuracy(Z_val, y_val, params, C, H)
            if acc > best_acc:
                best_acc = acc
                best_lr, best_l2 = lr, l2

    # Retrain on full data
    encoders_full, Z_full, stats_full = encode_features(X, types, fit=True)
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W1, b1, W2, b2 = train_mlp(Z_full, y, C, H, best_lr, best_l2, rng, epochs, BATCH)

    # Build lookup tables T
    W = H
    T_list = []
    off = 0
    for E in encoders_full:
        dim = E.shape[1]
        T_list.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # (d, 256, W)

    # Pack weights
    magic = 0x54424C31
    d = X.shape[1]
    header = struct.pack("<4i", magic, d, H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


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