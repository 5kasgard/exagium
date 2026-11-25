```hypothesis
We propose a new family "label-mlp-tuned" that extends the label-linear table-based encoding with a single ReLU hidden layer. The hidden size (4,8,16), learning rate, and L2 are chosen on the last 20% of train.in (tuning split), then the model is retrained on all data. The table-based first layer keeps inference fast (only d*256*W lookups and a small dense layer). We expect to beat label-linear's 0.9613 dev accuracy; if dev accuracy does not exceed 0.965, the added complexity is not justified.
```

```python
"""Table family with tuned one-hidden-layer MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encoding: per-byte ordinal z-scored, categorical one-hot over training values.
Hyperparameters (H, LR, L2) selected on last 20% of train.in (file order), then trained on all data.
Exported as per-byte lookup tables: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W];
if H: float32 W2[H][C], b2[C] (W = H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

# Fixed settings
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid
H_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fit_encoding(X, types):
    """Return list of encoding matrices (256 x dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoding(X, enc):
    """Transform X (n x d) using enc to Z (n x k)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (n x C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return (probs / n).astype(np.float32)


def forward_linear(Z, W, b):
    return Z @ W + b


def forward_mlp(Z, W1, b1, W2, b2):
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    return h @ W2 + b2, a, h


def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(Z_train, y_train, Z_val, y_val, H, C, LR, L2, seed):
    """Train on (Z_train, y_train), evaluate on (Z_val, y_val). Return (params, val_acc)."""
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
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            if H:
                logits, a, h = forward_mlp(z, *params)
                g = softmax_grad(logits, yy)
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * params[0],
                    gh.sum(0),
                    h.T @ g + L2 * params[2],
                    g.sum(0)
                ]
            else:
                logits = forward_linear(z, *params)
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + L2 * params[0],
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    if H:
        logits_val, _, _ = forward_mlp(Z_val, *params)
    else:
        logits_val = forward_linear(Z_val, *params)
    val_acc = accuracy(logits_val, y_val)
    return params, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_cfg = None
    best_enc = None
    best_params = None

    # Grid search
    cfg_idx = 0
    for H in H_CHOICES:
        for LR in LR_CHOICES:
            for L2 in L2_CHOICES:
                cfg_idx += 1
                # Fit encoding on training split only
                enc = fit_encoding(X_tr, types)
                Z_tr = apply_encoding(X_tr, enc)
                Z_val = apply_encoding(X_val, enc)
                # Train with a seed derived from config index for reproducibility
                params, val_acc = train_model(Z_tr, y_tr, Z_val, y_val, H, C, LR, L2, SEED + cfg_idx)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cfg = (H, LR, L2)
                    best_enc = enc
                    best_params = params
                print(f"Config H={H} LR={LR} L2={L2} -> val_acc={val_acc:.4f}", file=sys.stderr)

    print(f"Best config: H={best_cfg[0]} LR={best_cfg[1]} L2={best_cfg[2]} val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    H, LR, L2 = best_cfg
    enc_full = fit_encoding(X, types)
    Z_full = apply_encoding(X, enc_full)
    final_params, _ = train_model(Z_full, y, Z_full, y, H, C, LR, L2, SEED + 999)  # dummy val not used

    # Build weight file
    W = H if H else C
    # T: list of d matrices (256 x W)
    T = []
    off = 0
    for E in enc_full:
        # E is (256, dim_j), final_params[0] is (k, W) or (k, C)
        T.append((E @ final_params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    # b1
    b1 = final_params[1].astype(np.float32)
    # Pack
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = final_params[2].astype("<f4").tobytes()
        b2 = final_params[3].astype("<f4").tobytes()
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