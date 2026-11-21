```hypothesis
Propose a tuned one-hidden-layer MLP (table family) that selects hidden size H and optimization
hyperparameters on the tuning split (last 20% of train.in). The table architecture (per-position
byte lookup tables) is kept for fast inference; only H, learning rate, and L2 are tuned. This
should beat label-linear (H=0) by adding non-linearity and beat label-mlp16 (fixed H=16) by
using a larger, tuned H. If the best tuned model does not exceed label-linear's dev accuracy,
the hypothesis fails.
```

```python
"""Tuned table MLP for label tasks: chooses H and hyperparams on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on the tuning split
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_inputs(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def build_table(enc, W1, d):
    """Build per-position lookup table T[j][byte][w] = sum_{f} E_j[byte, f] * W1[offset+f, w]."""
    T = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    return np.stack(T)  # shape (d, 256, W)


def forward_table(T, b1, W2, b2, X, H):
    """Compute logits for a batch X (n x d) using table T."""
    n, d = X.shape
    W = H if H else T.shape[2]
    acc = np.tile(b1, (n, 1))  # (n, W)
    for j in range(d):
        acc += T[j, X[:, j]]   # (n, W)
    if H:
        h = np.maximum(acc, 0)  # ReLU
        logits = h @ W2 + b2
    else:
        logits = acc
    return logits


def train_model(X_tr, y_tr, X_val, y_val, enc, d, C, H, lr, l2, rng_seed):
    """Train on (X_tr, y_tr), return (T, b1, W2, b2, val_acc)."""
    k = sum(E.shape[1] for E in enc)
    rng = np.random.default_rng(rng_seed)
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
    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X_tr[idx], y_tr[idx]
            T = build_table(enc, W1, d)
            logits = forward_table(T, b1, W2, b2, z, H)
            g = softmax_grad(logits, yy)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                grads = [
                    z.T @ g + l2 * W1,
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # validation accuracy
    T = build_table(enc, W1, d)
    val_logits = forward_table(T, b1, W2, b2, X_val, H)
    val_pred = val_logits.argmax(axis=1)
    val_acc = (val_pred == y_val).mean()
    return T, b1, W2, b2, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    enc = encode_inputs(X_tr, types)  # encoding based on training split only

    best_acc = -1.0
    best_cfg = None
    best_weights = None

    # Grid search
    for i, H in enumerate(H_CHOICES):
        for j, lr in enumerate(LR_CHOICES):
            for k, l2 in enumerate(L2_CHOICES):
                # Deterministic seed per configuration
                cfg_seed = SEED + i * 100 + j * 10 + k
                T, b1, W2, b2, acc = train_model(X_tr, y_tr, X_val, y_val, enc, d, C, H, lr, l2, cfg_seed)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, lr, l2)
                    best_weights = (T, b1, W2, b2)
                print(f"  H={H}, lr={lr}, l2={l2} -> val_acc={acc:.4f}", file=sys.stderr)

    H_best, lr_best, l2_best = best_cfg
    print(f"Best: H={H_best}, lr={lr_best}, l2={l2_best}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config (fixed seed for reproducibility)
    enc_full = encode_inputs(X, types)
    T, b1, W2, b2, _ = train_model(X, y, X_val, y_val, enc_full, d, C, H_best, lr_best, l2_best, SEED)

    # Pack weights: magic, d, H, C; T[d][256][W]; b1[W]; if H: W2[H][C], b2[C]
    W = H_best if H_best else C
    out = struct.pack("<4i", 0x54424C31, d, H_best, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H_best:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
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