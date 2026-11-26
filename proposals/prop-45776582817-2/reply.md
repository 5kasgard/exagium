```hypothesis
A small one-hidden-layer MLP (32–128 units) with ReLU, trained with Adam and hyperparameters chosen on the tuning split (last 20% of train). The task benefits from non-linear interactions between context bytes; a linear model cannot capture them. The MLP adds modest inference cost (d·H + H·C ops) while improving accuracy. If dev accuracy does not exceed label-linear (0.3375) at similar or lower cost, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on the tuning split
GRID = [
    {"H": 32, "LR": 1e-3, "L2": 1e-4},
    {"H": 32, "LR": 3e-3, "L2": 1e-4},
    {"H": 32, "LR": 1e-2, "L2": 1e-4},
    {"H": 64, "LR": 1e-3, "L2": 1e-4},
    {"H": 64, "LR": 3e-3, "L2": 1e-4},
    {"H": 64, "LR": 1e-2, "L2": 1e-4},
    {"H": 128, "LR": 1e-3, "L2": 1e-4},
    {"H": 128, "LR": 3e-3, "L2": 1e-4},
    {"H": 128, "LR": 1e-2, "L2": 1e-4},
]
SEED_BASE = 1
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


def encode_inputs(X, card):
    """Build per-position encoding matrices and the dense design matrix Z."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z


def train_model(Z_train, y_train, Z_val, y_val, C, H, LR, L2, seed):
    """Train a one-hidden-layer MLP and return validation accuracy and parameters."""
    n, k = Z_train.shape
    rng = np.random.default_rng(seed)
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
            z, yy = Z_train[idx], y_train[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
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
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # Validation accuracy
    a_val = Z_val @ W1 + b1
    h_val = np.maximum(a_val, 0)
    logits_val = h_val @ W2 + b2
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params


def write_weights(card, enc, params, H, out_path):
    """Write weights in the table-family format expected by infer.c."""
    d = card["input"]["bytes"]
    C = card["classes"]
    W = H if H else C
    W1, b1, W2, b2 = params
    off = 0
    T = []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(out_path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n_total = X.shape[0]
    n_val = n_total // 5
    # Split: first 80% train, last 20% validation (file order)
    train_idx = np.arange(n_total - n_val)
    val_idx = np.arange(n_total - n_val, n_total)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Encode using training data only
    enc, Z_train = encode_inputs(X_train, card)
    Z_val = np.concatenate([E[X_val[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    best_acc = -1
    best_params = None
    best_H = 0
    best_config = None
    for i, config in enumerate(GRID):
        H = config["H"]
        LR = config["LR"]
        L2 = config["L2"]
        seed = SEED_BASE + i
        acc, params = train_model(Z_train, y_train, Z_val, y_val, C, H, LR, L2, seed)
        print(f"Config {config}: val acc = {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_H = H
            best_config = config
    print(f"Best config: {best_config}, val acc: {best_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset with best config
    enc_full, Z_full = encode_inputs(X, card)
    n, k = Z_full.shape
    H = best_H
    LR = best_config["LR"]
    L2 = best_config["L2"]
    rng = np.random.default_rng(SEED_BASE + 100)
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
            z, yy = Z_full[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
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
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    write_weights(card, enc_full, params, H, sys.argv[4])


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