```hypothesis
Family: label-mlp-tuned (small MLP with one hidden layer, hyperparameters tuned on validation split).
Reason: The 16-byte context for next-character prediction benefits from modeling interactions between positions, which a linear model cannot capture. A hidden layer of 32–128 units adds non-linearity with minimal inference overhead. Grid search over H, learning rate, and L2 on the last 20% of training data selects a robust configuration.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost per item exceeds ~1300 ns/item (2× label-linear).
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid (kept small for deterministic, fast tuning)
GRID_H = [64, 128]
GRID_LR = [3e-3]
GRID_L2 = [1e-4, 1e-5]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BASE_SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(card, X):
    """Build per-position one-hot encoders exactly like label-linear."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def build_Z(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_model(Z, y, H, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, seed):
    n, k = Z.shape
    C = int(y.max()) + 1
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
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
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
    return params

def eval_model(Z, y, params, H):
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

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    tune_size = int(0.2 * n)
    train_idx = np.arange(n - tune_size)
    val_idx = np.arange(n - tune_size, n)

    # Build encoders on full training data (as label-linear does)
    enc = build_encoders(card, X)
    Z_full = build_Z(X, enc)
    Z_train = Z_full[train_idx]
    Z_val = Z_full[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    best_acc = -1.0
    best_config = None
    config_idx = 0
    for H in GRID_H:
        for LR in GRID_LR:
            for L2 in GRID_L2:
                seed = BASE_SEED + config_idx
                config_idx += 1
                params = train_model(Z_train, y_train, H, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, seed)
                acc = eval_model(Z_val, y_val, params, H)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (H, LR, L2)
    print(f"Best config: H={best_config[0]}, LR={best_config[1]}, L2={best_config[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    H_best, LR_best, L2_best = best_config
    final_params = train_model(Z_full, y, H_best, LR_best, L2_best, BATCH, SAMPLES, MAX_EPOCHS, BASE_SEED)

    # Prepare weights for output (same format as label-linear)
    d = X.shape[1]
    C = card["classes"]
    W = H_best if H_best else C
    W1 = final_params[0]
    b1 = final_params[1]
    T_list = []
    off = 0
    for E in enc:
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # shape (d, 256, W)

    out = struct.pack("<4i", 0x54424C31, d, H_best, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H_best:
        W2 = final_params[2]
        b2 = final_params[3]
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