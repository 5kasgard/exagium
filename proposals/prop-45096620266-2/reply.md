```hypothesis
We propose a one-hidden-layer MLP (label-mlp) with hyperparameters (hidden size, learning rate, L2) selected on the tuning split (last 20% of train). The task is next-character prediction from 16 bytes; an MLP can capture interactions between context positions that a linear model cannot. The table-family inference format already supports a hidden layer, so we only need a new training procedure. If dev accuracy does not exceed label-linear (0.3375) by at least 0.01, the family does not justify its added complexity.
```

```python
"""Table family with one hidden layer and hyperparameter tuning on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
HIDDEN_SIZES = [32, 64, 128]
LEARNING_RATES = [1e-3, 3e-3]
L2S = [1e-4, 1e-3]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED_BASE = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x num_features) and Z (n x total_features)."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z

def train_mlp(Z_train, y_train, Z_val, y_val, H, lr, l2, batch, max_epochs, samples, seed):
    n, k = Z_train.shape
    C = 65
    rng = np.random.default_rng(seed)
    # Xavier initialization
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z_train[idx], y_train[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            gh = g @ W2.T
            gh[a <= 0] = 0
            grads = [
                z.T @ gh + l2 * W1,
                gh.sum(0),
                h.T @ g + l2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # validation accuracy
    a_val = Z_val @ W1 + b1
    h_val = np.maximum(a_val, 0)
    logits_val = h_val @ W2 + b2
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, (W1, b1, W2, b2)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Build encoding on the full training set (as the linear baseline does)
    enc, Z = build_encoding(X, card["input"]["types"])

    # Split: last 20% for tuning
    n_total = Z.shape[0]
    n_tune = n_total // 5
    train_idx = slice(0, n_total - n_tune)
    tune_idx = slice(n_total - n_tune, n_total)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    # Grid search
    best_acc = -1.0
    best_cfg = None
    best_params = None
    cfg_id = 0
    for H in HIDDEN_SIZES:
        for lr in LEARNING_RATES:
            for l2 in L2S:
                max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Z_train.shape[0]))
                seed = SEED_BASE + cfg_id
                cfg_id += 1
                acc, params = train_mlp(Z_train, y_train, Z_tune, y_tune,
                                        H, lr, l2, BATCH, max_epochs, SAMPLES, seed)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, lr, l2)
                    best_params = params

    # Retrain on full data with best config and fixed seed
    H, lr, l2 = best_cfg
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_total))
    _, (W1, b1, W2, b2) = train_mlp(Z, y, Z, y,  # validation not used
                                      H, lr, l2, BATCH, max_epochs, SAMPLES, SEED_BASE)

    # Build per-position lookup tables T (d x 256 x H)
    off = 0
    T = []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    # Write weights in table format
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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