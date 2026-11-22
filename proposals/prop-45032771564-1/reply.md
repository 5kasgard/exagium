```hypothesis
Family: label-mlp32 — a one-hidden-layer MLP with 32 ReLU units, using per-byte lookup tables for fast inference.
Why: The optdigits pixels have non-linear interactions; a small MLP captures these better than linear models while staying tiny (64*256*32 ≈ 2M parameters). The table-based inference keeps CPU cost low.
Failure: If dev accuracy ≤ label-linear (0.9613) or inference cost exceeds 2× label-linear.
```
```python
"""label-mlp32: one-hidden-layer MLP (32 ReLU units) with per-byte lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib+numpy only.
Hyperparameters (learning rate, L2) chosen on the last 20% of train.in (tuning split),
then final model trained on all data."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 32
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_enc(X, types):
    """Return list of encoding matrices E_j (256 x D_j) for each feature."""
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


def apply_enc(X, enc):
    """Transform raw byte matrix X (n x d) to concatenated encoded matrix Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_loop(Z, y, H, C, lr, l2, batch, samples, max_epochs, seed):
    """Train on encoded data Z (n x k), return params [W1, b1, W2, b2] (or [W1, b1] if H=0)."""
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            def softmax_grad(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + l2 * W1, gh.sum(0), h.T @ g + l2 * W2, g.sum(0)]
            else:
                g = softmax_grad(z @ W1 + b1)
                grads = [z.T @ g + l2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params, H, C):
    """Return accuracy on encoded data."""
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


def write_weights(enc, params, H, C, d, path):
    """Write weights.bin in the table format (magic, d, H, C, T, b1, [W2, b2])."""
    W = H if H else C
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    with open(path, "wb") as f:
        f.write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    types = card["input"]["types"]

    # Split: last 20% for tuning
    n = len(X)
    tune_size = int(0.2 * n)
    train_size = n - tune_size
    idx_train = slice(0, train_size)
    idx_tune = slice(train_size, n)

    X_train, y_train = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]

    # Hyperparameter search on tuning split
    best_acc = -1.0
    best_lr = LR_GRID[0]
    best_l2 = L2_GRID[0]
    for lr in LR_GRID:
        for l2 in L2_GRID:
            enc = compute_enc(X_train, types)
            Z_train = apply_enc(X_train, enc)
            Z_tune = apply_enc(X_tune, enc)
            params = train_loop(Z_train, y_train, H, C, lr, l2, BATCH, SAMPLES, MAX_EPOCHS, SEED)
            acc = evaluate(Z_tune, y_tune, params, H, C)
            if acc > best_acc:
                best_acc, best_lr, best_l2 = acc, lr, l2

    # Retrain on all data with best hyperparameters
    enc_full = compute_enc(X, types)
    Z_full = apply_enc(X, enc_full)
    final_params = train_loop(Z_full, y, H, C, best_lr, best_l2, BATCH, SAMPLES, MAX_EPOCHS, SEED)
    write_weights(enc_full, final_params, H, C, d, sys.argv[4])


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
        fprintf(stderr, "bad weights size\n"); return 3; }
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