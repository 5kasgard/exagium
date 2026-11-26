```hypothesis
Family: label-mlp32 – a one-hidden-layer MLP with 32 ReLU units using per-pixel z-score tables.
Why: Optdigits pixels (0–16) have non-linear interactions that a linear model misses; a small hidden layer captures them cheaply.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-linear, the family does not justify itself.
```

```python
"""label-mlp32: 64 ordinal pixels -> z-score tables -> 32 ReLU -> 10 logits.
Train: tune learning rate on last 20% of train.in (file order), then retrain on all data.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic, seed=1, single thread)"""
import json, math, struct, sys

import numpy as np

H = 32
SEED = 1
LR_CANDIDATES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS_TUNE = 100
MAX_EPOCHS_FULL = 200


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(X, types, fit=True, stats=None):
    """Return (Z, new_stats). Z is float32 [n, k]. stats is list per feature of (mean,std) or (vals,E)."""
    n, d = X.shape
    if fit:
        stats = []
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            if fit:
                mu, sd = v.mean(), v.std() or 1.0
                stats.append((mu, sd))
            else:
                mu, sd = stats[j]
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
        else:
            if fit:
                vals = np.unique(X[:, j])
                E = np.zeros((256, len(vals)), np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                stats.append((vals, E))
            else:
                vals, E = stats[j]
        enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return Z, stats


def train_model(Ztr, ytr, Zval, yval, lr, epochs, rng):
    n, k = Ztr.shape
    C = 10
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Ztr[idx], ytr[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    if Zval is not None:
        a = Zval @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(1)
        acc = (pred == yval).mean()
        return acc, params
    return None, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    n = len(X)
    split = int(0.8 * n)
    Xtr, Xval = X[:split], X[split:]
    ytr, yval = y[:split], y[split:]

    # Tune learning rate
    rng = np.random.default_rng(SEED)
    Ztr, stats = encode(Xtr, types, fit=True)
    Zval, _ = encode(Xval, types, fit=False, stats=stats)
    best_lr, best_acc = LR_CANDIDATES[0], -1.0
    for lr in LR_CANDIDATES:
        acc, _ = train_model(Ztr, ytr, Zval, yval, lr, MAX_EPOCHS_TUNE, rng)
        if acc > best_acc:
            best_acc, best_lr = acc, lr

    # Retrain on all data
    Zall, stats = encode(X, types, fit=True)
    rng = np.random.default_rng(SEED)
    _, params = train_model(Zall, y, None, None, best_lr, MAX_EPOCHS_FULL, rng)
    W1, b1, W2, b2 = params

    # Build per-pixel tables T[d][256][H]
    off, T = 0, []
    for j, t in enumerate(types):
        E = np.zeros((256, Zall.shape[1] // len(types)), np.float32)  # wrong, need per-feature dim
    # Recompute encoding matrices per feature
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            mu, sd = stats[j]
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
        else:
            vals, E = stats[j]
        enc.append(E)
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, 10)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (one ReLU layer) for label tasks: ./infer weights.bin in.rec N
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