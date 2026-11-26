```hypothesis
Family: label-mlp2l (two-layer MLP with feature-wise embedding tables). The adult task has mixed ordinal/categorical features; a two-layer ReLU net (H1=32, H2=16) can capture pairwise interactions that linear models miss, while the per-feature embedding tables keep inference fast. Tuning H1/H2 on the last 20% of train.in should yield a model that beats label-linear's 0.8524 accuracy. If dev accuracy ≤ 0.855, the extra capacity does not justify the cost.
```
```python
"""Table family: two-layer MLP (H1, H2) with per-feature embedding tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored (1 dim), categorical -> one-hot over training values.
Hyperparameters (H1, H2) chosen on tuning split (last 20% of train.in), then retrained on all train.
Exported weights: magic 0x54424C32, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

# Adam hyperparameters (same as label-linear)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# Grid to search on tuning split (H1, H2)
GRID = [(16, 8), (32, 16), (64, 32)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x D_j) and total dim D."""
    enc, D = [], 0
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # 256 x 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float64)
            E[vals, np.arange(len(vals))] = 1.0                # 256 x |vals|
        enc.append(E)
        D += E.shape[1]
    return enc, D

def train_mlp(Z, y, H1, H2, C, rng, n_samples, n_features):
    """Train two-layer MLP on Z (n x D) with widths H1, H2. Return params."""
    D = Z.shape[1]
    # He init
    W1 = (rng.uniform(-1, 1, (D, H1)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_samples))
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for s in range(0, n_samples, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            a3 = h2 @ W3 + b3
            g3 = softmax_grad(a3, yy)
            # backward
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(0)
            g2 = (g3 @ W3.T) * (a2 > 0)
            grad_W2 = h1.T @ g2 + L2 * W2
            grad_b2 = g2.sum(0)
            g1 = (g2 @ W2.T) * (a1 > 0)
            grad_W1 = z.T @ g1 + L2 * W1
            grad_b1 = g1.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def eval_mlp(Z, y, params):
    W1, b1, W2, b2, W3, b3 = params
    a1 = Z @ W1 + b1
    h1 = np.maximum(a1, 0)
    a2 = h1 @ W2 + b2
    h2 = np.maximum(a2, 0)
    a3 = h2 @ W3 + b3
    pred = a3.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    # build encoding on full train (as per card: choose settings on tuning split, then train on all)
    enc, D = build_encoding(X, types)
    Z_tr = np.concatenate([E[X_tr[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    Z_tune = np.concatenate([E[X_tune[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_H1, best_H2 = -1, None, None, None
    for H1, H2 in GRID:
        params = train_mlp(Z_tr, y_tr, H1, H2, C, rng, split, D)
        acc = eval_mlp(Z_tune, y_tune, params)
        if acc > best_acc:
            best_acc, best_params, best_H1, best_H2 = acc, params, H1, H2

    # retrain on full train with best H1, H2
    Z_full = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    rng = np.random.default_rng(SEED)  # reset seed for determinism
    final_params = train_mlp(Z_full, y, best_H1, best_H2, C, rng, n, D)

    # convert to per-feature tables
    W1, b1, W2, b2, W3, b3 = final_params
    off = 0
    T1 = []
    for E in enc:
        T1.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    d = len(enc)
    # pack weights
    magic = 0x54424C32
    out = struct.pack("<4i", magic, d, best_H1, best_H2, C)
    out += np.stack(T1).astype("<f4").tobytes()          # d * 256 * H1
    out += b1.astype("<f4").tobytes()                    # H1
    out += W2.astype("<f4").tobytes()                    # H1 * H2
    out += b2.astype("<f4").tobytes()                    # H2
    out += W3.astype("<f4").tobytes()                    # H2 * C
    out += b3.astype("<f4").tobytes()                    # C
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Table family: two-layer MLP (H1, H2) with per-feature embedding tables.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 20 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < H1; u++) acc1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        // ReLU layer 1
        for (int u = 0; u < H1; u++) if (acc1[u] < 0.0f) acc1[u] = 0.0f;
        // layer 2
        for (int v = 0; v < H2; v++) acc2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = acc1[u];
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        // ReLU layer 2
        for (int v = 0; v < H2; v++) if (acc2[v] < 0.0f) acc2[v] = 0.0f;
        // output layer
        for (int c = 0; c < C; c++) acc3[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = acc2[v];
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) acc3[c] += hv * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```