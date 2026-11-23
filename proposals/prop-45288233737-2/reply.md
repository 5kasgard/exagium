```hypothesis
We propose "label-mlp-tuned": a one-hidden-layer ReLU MLP on per-position one-hot byte embeddings, with hidden size and regularization chosen on the tuning split (last 20% of train.in). This captures non-linear context interactions that the linear model misses, and should exceed label-linear's 0.3375 dev accuracy while keeping inference fast (similar to label-mlp16 but with better hyperparameters). If the best tuned model does not beat label-linear on dev, the family fails.
```

```python
"""label-mlp-tuned: one-hidden-layer ReLU MLP with hyperparameter tuning on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed seeds for reproducibility
SEED_BASE = 12345
# Hyperparameter grid (kept small to stay within time limits)
H_VALUES = [0, 16, 32, 64]
LR_VALUES = [3e-3]
L2_VALUES = [1e-4, 1e-3]
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


def build_encoding(X, d):
    """Return list of (256, num_unique) one-hot matrices for each position."""
    enc = []
    for j in range(d):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    return enc


def encode(X, enc, d):
    """Convert X (n, d) to concatenated one-hot matrix (n, total_features)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Gradient of cross-entropy w.r.t. logits."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_eval(X_tr, y_tr, X_val, y_val, enc, d, C, H, lr, l2, seed):
    """Train on (X_tr, y_tr), return validation accuracy."""
    rng = np.random.default_rng(seed)
    Z_tr = encode(X_tr, enc, d)
    Z_val = encode(X_val, enc, d)
    n, k = Z_tr.shape
    W = H if H else C

    # Initialize parameters
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
            z, yy = Z_tr[idx], y_tr[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
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

    # Validation accuracy
    if H:
        a = Z_val @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z_val @ W1 + b1
    preds = logits.argmax(axis=1)
    return float((preds == y_val).mean())


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)

    # Tuning split: last 20% in file order
    split = int(0.8 * n_all)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Build encoding on the training portion (for hyperparameter search)
    enc = build_encoding(X_tr, d)

    best_acc = -1.0
    best_cfg = None
    cfg_idx = 0
    for H in H_VALUES:
        for lr in LR_VALUES:
            for l2 in L2_VALUES:
                seed = SEED_BASE + cfg_idx
                acc = train_eval(X_tr, y_tr, X_val, y_val, enc, d, C, H, lr, l2, seed)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, lr, l2, seed)
                cfg_idx += 1

    # Retrain on all data with best config
    H, lr, l2, seed = best_cfg
    enc_full = build_encoding(X_all, d)
    Z_all = encode(X_all, enc_full, d)
    n, k = Z_all.shape
    W = H if H else C
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
            z, yy = Z_all[idx], y_all[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
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

    # Export weights in the same format as label-linear (magic, d, H, C, T, b1, [W2, b2])
    off, T = 0, []
    for E in enc_full:
        T.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T).astype("<f4")
    b1 = b1.astype("<f4")
    out = struct.pack("<4i", 0x54424C31, d, H, C) + T.tobytes() + b1.tobytes()
    if H:
        out += params[2].astype("<f4").tobytes() + params[3].astype("<f4").tobytes()
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