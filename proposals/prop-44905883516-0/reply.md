```hypothesis
Family: label-mlp8 – a one-hidden-layer MLP with 8 ReLU units (W=8) trained with Adam.
Why: Adult has non-linear feature interactions; 8 units capture enough non-linearity to beat linear (0.8524) while being ~2x faster than mlp16 (H=16). Tuning learning rate on the last 20% of train.in ensures good generalization.
Failure: dev accuracy ≤ 0.8524 or inference cost > 2325 Ir/item (mlp16 cost).
```

```python
"""label-mlp8: one-hidden-layer MLP with 8 ReLU units, tuned on last 20% of train.in.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, math, struct, sys

import numpy as np

H = 8
SEED = 1
LR_CANDIDATES = [1e-3, 3e-3, 1e-2]
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


def make_encoders(X, types):
    """Return list of (256, feat_dim) float64 encoder matrices, one per column."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoders(X, encoders):
    """X (n, d) uint8 -> Z (n, total_feat) float32."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(encoders)], axis=1).astype(np.float32)


def init_params(k, C, rng):
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]


def sg(S, yy):
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(yy)), yy] -= 1.0
    return (P / len(yy)).astype(np.float32)


def train_epoch(Z, y, params, state, rng, lr, t):
    n = Z.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        z, yy = Z[idx], y[idx]
        if H:
            a = z @ params[0] + params[1]
            h = np.maximum(a, 0)
            g = sg(h @ params[2] + params[3], yy)
            gh = (g @ params[2].T) * (a > 0)
            grads = [
                z.T @ gh + L2 * params[0],
                gh.sum(0),
                h.T @ g + L2 * params[2],
                g.sum(0),
            ]
        else:
            g = sg(z @ params[0] + params[1], yy)
            grads = [z.T @ g + L2 * params[0], g.sum(0)]
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return t


def accuracy(Z, y, params):
    if H:
        a = Z @ params[0] + params[1]
        h = np.maximum(a, 0)
        scores = h @ params[2] + params[3]
    else:
        scores = Z @ params[0] + params[1]
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    n = X_all.shape[0]
    val_size = n // 5
    tr_idx = slice(0, n - val_size)
    va_idx = slice(n - val_size, n)

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]

    best_lr = LR_CANDIDATES[0]
    best_acc = -1.0

    for trial_idx, lr in enumerate(LR_CANDIDATES):
        rng = np.random.default_rng(SEED + trial_idx)
        encoders = make_encoders(X_tr, types)
        Z_tr = apply_encoders(X_tr, encoders)
        Z_va = apply_encoders(X_va, encoders)
        k = Z_tr.shape[1]

        params = init_params(k, C, rng)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Z_tr.shape[0]))
        for _ in range(epochs):
            t = train_epoch(Z_tr, y_tr, params, state, rng, lr, t)
        acc = accuracy(Z_va, y_va, params)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    # Final training on all data with best_lr
    rng = np.random.default_rng(SEED)
    encoders = make_encoders(X_all, types)
    Z_all = apply_encoders(X_all, encoders)
    k = Z_all.shape[1]

    params = init_params(k, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Z_all.shape[0]))
    for _ in range(epochs):
        t = train_epoch(Z_all, y_all, params, state, rng, best_lr, t)

    # Build per-byte lookup tables T
    off, T = 0, []
    for E in encoders:
        T.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<4i", 0x54424C31, X_all.shape[1], H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2

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