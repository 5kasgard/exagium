```hypothesis
We propose a small one-hidden-layer MLP with 8 ReLU units (label-mlp8) using per-pixel z-score normalization.
The 8x8 optdigits pixels have simple spatial structure; a tiny non-linear layer should capture interactions
that linear models miss, while keeping inference cost near label-linear (≈600 MACs/item).
Training uses Adam with hyperparameters chosen on the last 20% of train.in (tuning split), then refits on all data.
Expected dev accuracy >0.965 (beating label-linear's 0.9613) at ≤6000 Ir/item.
If dev accuracy ≤0.962, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8
SEED = 1

# Hyperparameter grid for tuning split
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
BATCH = 64
MAX_EPOCHS_TUNE = 50
MAX_EPOCHS_FULL = 100
SAMPLES = 600000  # for epoch cap


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices (256 x dim_j) and total dim."""
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


def encode_data(X, enc):
    """X (n, d) -> Z (n, total_dim) using precomputed enc."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def make_lookup(enc, W1):
    """Build per-feature lookup tables T[j][v] = enc_j[v] @ W1_slice."""
    T = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    return T


def train_model(Z, y, C, H, lr, l2, epochs, batch, rng):
    n, k = Z.shape
    # Xavier init for W1, He for W2
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            g = softmax_grad(h @ W2 + b2, yy)
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


def evaluate(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(card, enc, params):
    d = card["input"]["bytes"]
    C = card["classes"]
    W1, b1, W2, b2 = params
    T = make_lookup(enc, W1)
    # magic, d, H, C
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    # T: d x 256 x H
    out += np.stack(T).astype("<f4").tobytes()
    # b1: H
    out += b1.astype("<f4").tobytes()
    # W2: H x C
    out += W2.astype("<f4").tobytes()
    # b2: C
    out += b2.astype("<f4").tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    # No shuffling for deterministic split (file order)
    train_idx, tune_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Build encoding on full training set (to avoid data leakage, use only train_idx for stats? 
    # But card says choose settings on tuning split, then train on all of train.
    # We'll compute encoding stats on the full train.in (as label-linear does) because the test gate
    # expects the same encoding as the final model. However, for tuning we should use encoding
    # derived from the training portion only? The contract: "choose them on the card's tuning split
    # only (last 20% of train.in in file order), then train on all of train". 
    # It doesn't forbid using the whole train.in for preprocessing. To be safe, we compute
    # encoding on the full train.in (as the final model will), but for tuning we only train on
    # the first 80% and validate on the last 20%. This is acceptable because the tuning split
    # is only used to choose hyperparameters, not to fit the model.
    enc = build_encoding(X, types)
    Z_full = encode_data(X, enc)
    Z_train = Z_full[train_idx]
    Z_tune = Z_full[tune_idx]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_lr = LR_GRID[0]
    best_l2 = L2_GRID[0]

    for lr in LR_GRID:
        for l2 in L2_GRID:
            params = train_model(Z_train, y_train, C, H, lr, l2, MAX_EPOCHS_TUNE, BATCH, rng)
            acc = evaluate(Z_tune, y_tune, params)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_lr, best_l2 = lr, l2

    # Retrain on full data with best hyperparameters
    rng_full = np.random.default_rng(SEED)  # reset seed for deterministic final training
    final_params = train_model(Z_full, y, C, H, best_lr, best_l2, MAX_EPOCHS_FULL, BATCH, rng_full)

    # Pack and write
    out = pack_weights(card, enc, final_params)
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