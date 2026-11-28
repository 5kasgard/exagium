```hypothesis
Family: label-mlp-tune (one-hidden-layer MLP with validation-chosen width).
Why: The Adult dataset has non-linear feature interactions; a small MLP can capture them. Tuning hidden size on the last 20% of train.in (per card) picks the best capacity without overfitting. Beats label-linear (0.8524) and matches label-mlp16 (0.8591) with ≤16 units → lower CPU cost.
Failure: If validation accuracy never exceeds linear baseline, or final dev accuracy < 0.8524.
```

```python
"""label-mlp-tune: one-hidden-layer MLP with validation-tuned hidden size.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encoding: ordinal z-scored, categorical one-hot over training values.
Hyperparameter grid: hidden_size in [8, 16, 32], lr in [1e-3, 3e-3, 1e-2].
Validation split: last 20% of train.in (file order). Best config retrained on all train.
Exported weights format identical to label-linear (magic 0x54424C31, d, H, C, T, b1, W2, b2)."""
import json, math, struct, sys

import numpy as np

SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES_FULL = 600000
MAX_EPOCHS_FULL = 100
SAMPLES_SEARCH = 300000
MAX_EPOCHS_SEARCH = 50
HIDDEN_GRID = [8, 16, 32]
LR_GRID = [1e-3, 3e-3, 1e-2]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices (256 x dim) for each column."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoding(X, enc):
    """X (n, d) -> Z (n, total_dim) using precomputed enc list."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_mlp(Z, y, H, C, lr, samples, max_epochs, rng):
    """Train MLP on given data, return params (W1, b1, W2, b2)."""
    n, k = Z.shape
    W = H if H else C
    # init
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            if H:
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = a
            # softmax grad
            logits = logits - logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # backward
            if H:
                gh = g @ W2.T
                gh *= (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0)
                ]
            else:
                grads = [
                    z.T @ g + L2 * W1,
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


def evaluate(Z, y, params, H, C):
    W1, b1 = params[0], params[1]
    if H:
        W2, b2 = params[2], params[3]
    a = Z @ W1 + b1
    if H:
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = a
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(card, enc, params, H, C):
    """Serialize to the standard format."""
    W = H if H else C
    # T: per-column lookup tables (d, 256, W)
    off, T_list = 0, []
    for E in enc:
        T_list.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)
    out = struct.pack("<4i", 0x54424C31, card["input"]["bytes"], H, C)
    out += T.astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = len(X)
    split = int(0.8 * n)
    # validation split indices (last 20%)
    idx_val = np.arange(split, n)
    idx_tr = np.arange(split)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    # Build encoding on training split for search
    enc_tr = build_encoding(X_tr, types)
    Z_tr = apply_encoding(X_tr, enc_tr)
    Z_val = apply_encoding(X_val, enc_tr)

    best_acc = -1.0
    best_H = HIDDEN_GRID[0]
    best_lr = LR_GRID[0]
    best_params = None
    rng_base = np.random.default_rng(SEED)

    for H in HIDDEN_GRID:
        for lr in LR_GRID:
            rng = np.random.default_rng(rng_base.integers(2**32))
            params = train_mlp(Z_tr, y_tr, H, C, lr, SAMPLES_SEARCH, MAX_EPOCHS_SEARCH, rng)
            acc = evaluate(Z_val, y_val, params, H, C)
            if acc > best_acc:
                best_acc = acc
                best_H = H
                best_lr = lr
                best_params = params

    # Retrain on full data with best config
    enc_full = build_encoding(X, types)
    Z_full = apply_encoding(X, enc_full)
    rng_final = np.random.default_rng(SEED)
    final_params = train_mlp(Z_full, y, best_H, C, best_lr, SAMPLES_FULL, MAX_EPOCHS_FULL, rng_final)

    # Write weights
    weights = pack_weights(card, enc_full, final_params, best_H, C)
    open(sys.argv[4], "wb").write(weights)


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