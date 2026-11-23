```hypothesis
We propose "label-small-mlp": a one-hidden-layer MLP with hidden size tuned from {4,8,16} on the tuning split.
Adult's mixed ordinal/categorical features have non-linear interactions that a tiny MLP captures with minimal overhead.
If dev accuracy ≤ label-linear (0.8524) or inference cost ≥ label-mlp16 (2325 Ir/item), the family fails.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid
H_GRID = [4, 8, 16]
LR_GRID = [1e-3, 3e-3]
L2_GRID = [1e-4, 1e-3]

# Fixed training settings
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def get_enc(X, types):
    """Return list of encoding matrices (256, dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    return enc

def apply_enc(X, enc):
    """Apply encoding matrices to X, return concatenated float32 matrix."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)

def train_model(Z_train, y_train, Z_val, y_val, H, LR, L2, epochs, batch, seed, C):
    """Train MLP, return (params, val_acc). params = (W1, b1, W2, b2) if H>0 else (W1, b1)."""
    n, k = Z_train.shape
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
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z_train[idx], y_train[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # Validation accuracy
    if H:
        a = Z_val @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z_val @ W1 + b1
    pred = logits.argmax(axis=1)
    val_acc = (pred == y_val).mean()
    return params, val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    tune_size = n // 5
    train_size = n - tune_size
    X_train, X_tune = X[:train_size], X[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    # Hyperparameter search
    best_acc = -1.0
    best_params = None
    best_H = None
    best_LR = None
    best_L2 = None
    for H in H_GRID:
        for LR in LR_GRID:
            for L2 in L2_GRID:
                enc = get_enc(X_train, types)
                Z_train = apply_enc(X_train, enc)
                Z_tune = apply_enc(X_tune, enc)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_size))
                params, val_acc = train_model(Z_train, y_train, Z_tune, y_tune, H, LR, L2, epochs, BATCH, SEED, C)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = params
                    best_H = H
                    best_LR = LR
                    best_L2 = L2
    print(f"Best: H={best_H}, LR={best_LR}, L2={best_L2}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    enc_full = get_enc(X, types)
    Z_full = apply_enc(X, enc_full)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params_full, _ = train_model(Z_full, y, Z_full, y, best_H, best_LR, best_L2, epochs_full, BATCH, SEED, C)
    W1, b1 = params_full[0], params_full[1]
    if best_H:
        W2, b2 = params_full[2], params_full[3]
    else:
        W2, b2 = None, None

    # Build per-byte lookup tables T[j][v] = E[v] @ W1[offset:offset+dim_j]
    W = best_H if best_H else C
    off = 0
    T_list = []
    for E in enc_full:
        dim = E.shape[1]
        Tj = (E.astype(np.float32) @ W1[off:off+dim].astype(np.float32)).astype(np.float32)  # (256, W)
        T_list.append(Tj)
        off += dim
    T = np.stack(T_list)  # (d, 256, W)

    # Write weights.bin
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
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