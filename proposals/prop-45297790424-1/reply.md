```hypothesis
Propose a small MLP with one hidden layer of 8 ReLU units (label-mlp8), using the same per-pixel lookup-table encoding as label-linear. The non-linearity should capture pixel interactions that a linear model misses, improving accuracy over label-linear (0.9613) while keeping inference cost low (~6000 Ir/item). If dev accuracy does not exceed 0.9613, the hypothesis fails.
```

```python
"""label-mlp8: small MLP with 8 ReLU units for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal features by z-scoring per pixel (256 lookup values), then trains
a one-hidden-layer MLP (H=8) with Adam. Hyperparameters tuned on last 20% of train.
Exports weights in the same table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_CANDIDATES = [4, 8, 16]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit=True, stats=None):
    """Return Z (n x k) and list of encoder tables E_j (256 x enc_dim_j)."""
    n, d = X.shape
    if fit:
        stats = []
    encoders = []
    off = 0
    Z_cols = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            if fit:
                mu, sd = v.mean(), v.std() or 1.0
                stats.append((mu, sd))
            else:
                mu, sd = stats[j]
            E = ((np.arange(256) - mu) / sd)[:, None]  # 256 x 1
            encoders.append(E)
            Z_cols.append(E[X[:, j]])  # n x 1
        else:  # categorical (not used in optdigits)
            if fit:
                vals = np.unique(X[:, j])
                stats.append(vals)
            else:
                vals = stats[j]
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            encoders.append(E)
            Z_cols.append(E[X[:, j]])
    Z = np.concatenate(Z_cols, axis=1).astype(np.float32)
    return Z, encoders, stats


def train_mlp(X, y, H, seed):
    """Train MLP on given data, return params (W1,b1,W2,b2) and encoders."""
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    C = card["classes"]
    Z, encoders, stats = encode_features(X, types, fit=True)
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    # Xavier init
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # dL/dlogits
            # backprop
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh[a <= 0] = 0.0  # ReLU gradient
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params, encoders, stats


def evaluate(X, y, params, encoders, stats, types):
    Z, _, _ = encode_features(X, types, fit=False, stats=stats)
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def build_weight_blob(card, encoders, params):
    """Pack weights into the same binary format as label-linear."""
    d = card["input"]["bytes"]
    H = params[0].shape[1]  # W1 shape (k, H)
    C = card["classes"]
    W = H
    # Compute T tables: for each feature j, T_j = E_j @ W1[off:off+enc_dim]
    T_list = []
    off = 0
    for E in encoders:
        enc_dim = E.shape[1]
        T_j = (E @ params[0][off:off + enc_dim]).astype(np.float32)  # 256 x W
        T_list.append(T_j)
        off += enc_dim
    T = np.stack(T_list)  # d x 256 x W
    b1 = params[1].astype(np.float32)
    W2 = params[2].astype(np.float32)
    b2 = params[3].astype(np.float32)
    blob = struct.pack("<4i", 0x54424C31, d, H, C)
    blob += T.astype("<f4").tobytes()
    blob += b1.astype("<f4").tobytes()
    blob += W2.astype("<f4").tobytes()
    blob += b2.astype("<f4").tobytes()
    return blob


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    # Split: last 20% for tuning (file order)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_H = H_CANDIDATES[0]
    best_acc = -1.0
    for H in H_CANDIDATES:
        params, encoders, stats = train_mlp(X_train, y_train, H, SEED + H)
        acc = evaluate(X_tune, y_tune, params, encoders, stats, card["input"]["types"])
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on full data with best_H
    params, encoders, stats = train_mlp(X, y, best_H, SEED + best_H)
    blob = build_weight_blob(card, encoders, params)
    open(sys.argv[4], "wb").write(blob)


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