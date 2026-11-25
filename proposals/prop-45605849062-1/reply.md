```hypothesis
Family: label-mlp8 (one-hidden-layer ReLU network with 8 hidden units).
Why: The optdigits task is non-linear; label-linear (logistic regression) underfits at 0.9613. label-mlp16 (16 hidden units) reaches 0.9770 but costs ~70% more cycles. A smaller hidden layer (8 units) should retain most of the non-linear benefit while cutting inference cost (matrix multiplies scale with H). Hyperparameters (H, LR) are chosen on the tuning split (last 20% of train) to avoid overfitting.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ label-mlp16, the hypothesis is disproved.
```

```python
"""label-mlp8: one-hidden-layer ReLU network with 8 hidden units, tuned on validation split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid to search on the tuning split (last 20% of train)
H_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
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


def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def build_design(X, enc):
    """X: (n, d) uint8, enc: list of (256, dim_j) -> Z: (n, k) float32"""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """logits (n, C), y (n,) -> gradient (n, C) float32"""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(Z, y, H, C, LR, rng):
    """Train one-hidden-layer ReLU net (or linear if H=0) with Adam.
    Returns params: [W1, b1, W2, b2] if H>0 else [W1, b1]."""
    n, k = Z.shape
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32); b1 = np.zeros(C, np.float32)
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
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0),
                         h.T @ g + L2 * W2, g.sum(0)]
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def eval_accuracy(Z, y, params, H, C):
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(enc, params, H, C, d):
    """Serialise to the same format as label-linear infer.c expects."""
    # Build per-byte lookup tables T_j for the first layer (size d x 256 x W)
    W = H if H else C
    W1 = params[0]  # (k, W)
    off = 0
    T_list = []
    for E in enc:
        dim_j = E.shape[1]
        T_list.append((E.astype(np.float32) @ W1[off:off + dim_j]).astype(np.float32))
        off += dim_j
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        out += W2 + b2
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    enc = encode_features(X, types)  # use full data stats for encoding (as in label-linear)
    Z_train = build_design(X_train, enc)
    Z_tune = build_design(X_tune, enc)

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for H in H_CHOICES:
        for LR in LR_CHOICES:
            rng = np.random.default_rng(SEED)
            params = train_model(Z_train, y_train, H, C, LR, rng)
            acc = eval_accuracy(Z_tune, y_tune, params, H, C)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (H, LR)
                best_params = params

    # Retrain on full training set with best config
    H_best, LR_best = best_cfg
    rng = np.random.default_rng(SEED)
    Z_full = build_design(X, enc)
    final_params = train_model(Z_full, y, H_best, C, LR_best, rng)

    # Write weights
    weights = pack_weights(enc, final_params, H_best, C, d)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Format: see train.py. Identical to label-linear infer.c, supports H>0. */
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