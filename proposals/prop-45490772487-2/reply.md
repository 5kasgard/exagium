```hypothesis
We propose label-mlp64: a one-hidden-layer MLP with 64 ReLU units using per-byte lookup tables (same format as label-linear). The hidden layer captures byte interactions crucial for character prediction. Hyperparameters (hidden size, learning rate, L2) are selected on the last 20% tuning split, then the model is retrained on all data. Expected to beat label-linear (0.3375) with similar inference cost (~7k Ir/item). Failure if dev accuracy ≤ 0.3375 or inference cost > 10k Ir/item.
```

```python
"""label-mlp64: one-hidden-layer MLP with hyperparameter tuning on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed settings
SEED = 1
BATCH = 64
SAMPLES_TUNE = 200000
MAX_EPOCHS_TUNE = 50
SAMPLES_FULL = 600000
MAX_EPOCHS_FULL = 100

# Hyperparameter grid
H_GRID = [32, 64, 128]
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-5]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(X, types):
    """Return list of encoder matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode(X, enc):
    """Apply encoders to X (n x d) -> Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def init_params(k, H, C, rng):
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

def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)

def forward(z, params, H):
    if H:
        a = z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
        return logits, a, h
    else:
        logits = z @ params[0] + params[1]
        return logits, None, None

def backward(z, y, params, H, a, h, L2):
    g = softmax_grad(a @ params[2] + params[3] if H else z @ params[0] + params[1], y)
    if H:
        gh = (g @ params[2].T) * (a > 0)
        grads = [
            z.T @ gh + L2 * params[0],
            gh.sum(0),
            h.T @ g + L2 * params[2],
            g.sum(0)
        ]
    else:
        grads = [
            z.T @ g + L2 * params[0],
            g.sum(0)
        ]
    return grads

def train_loop(Z, y, params, H, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, rng):
    n = Z.shape[0]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits, a, h = forward(z, params, H)
            grads = backward(z, yy, params, H, a, h, L2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def accuracy(Z, y, params, H):
    logits, _, _ = forward(Z, params, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Build encoders on full training set
    enc = build_encoders(X, card["input"]["types"])
    Z_full = encode(X, enc)
    k = Z_full.shape[1]

    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    Z_train, y_train = Z_full[:split], y[:split]
    Z_val, y_val = Z_full[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search
    for H in H_GRID:
        for LR in LR_GRID:
            for L2 in L2_GRID:
                params = init_params(k, H, C, rng)
                params = train_loop(Z_train, y_train, params, H, LR, L2, BATCH,
                                    SAMPLES_TUNE, MAX_EPOCHS_TUNE, rng)
                acc = accuracy(Z_val, y_val, params, H)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, LR, L2)
                    best_params = [p.copy() for p in params]

    print(f"Best config: H={best_cfg[0]}, LR={best_cfg[1]}, L2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    H, LR, L2 = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for deterministic final training
    params = init_params(k, H, C, rng)
    params = train_loop(Z_full, y, params, H, LR, L2, BATCH,
                        SAMPLES_FULL, MAX_EPOCHS_FULL, rng)

    # Build lookup tables T for inference
    off = 0
    T_list = []
    for E in enc:
        W = H if H else C
        T_list.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # d x 256 x W
    b1 = params[1].astype(np.float32)

    # Write weights.bin
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        out += W2 + b2
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