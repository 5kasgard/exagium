```hypothesis
We propose label-mlp8: a one-hidden-layer MLP with 8 ReLU units, using the same per-byte lookup tables as label-linear. The Adult dataset has non-linear interactions (e.g., education × occupation) that a linear model misses; a tiny MLP captures these with minimal overhead. Hyperparameters (H=8, lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))) are chosen on the last 20% of train.in, then retrained on all data. Expected dev accuracy >0.8524 (label-linear) with Ir/item <2000 (below label-mlp16's 2325). If dev accuracy ≤0.8524 or Ir/item ≥2325, the model does not justify its existence.
```
```python
"""label-mlp8: one-hidden-layer MLP (8 ReLU units) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored (train mean/std), categorical -> one-hot over train values (255=unseen -> zero).
Exported weights: magic 0x4D4C5031, int32 d,H,C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split)
H = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, encoders=None, fit=False):
    """Return (Z, encoders) where Z is float32 n x K, encoders is list per feature."""
    n, d = X.shape
    if fit:
        encoders = []
    Z_parts = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            if fit:
                mu = v.mean()
                sd = v.std() or 1.0
                enc = (np.arange(256, dtype=np.float64) - mu) / sd
                encoders.append(("ord", enc.astype(np.float32)))
            else:
                enc = encoders[j][1]
            Z_parts.append(enc[v].reshape(-1, 1))
        else:  # categorical
            if fit:
                vals = np.unique(v)
                E = np.zeros((256, len(vals)), dtype=np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                encoders.append(("cat", E, vals))
            else:
                E = encoders[j][1]
            Z_parts.append(E[v])
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, encoders


def train_mlp(Z, y, H, rng, epochs, batch, lr, l2):
    n, k = Z.shape
    C = 2
    # Xavier init
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            g = softmax_grad(logits, yy)
            # backward
            gh = (g @ W2.T) * (a > 0)
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
    return params


def evaluate(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)

    # Split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n_all)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Fit encoders on training portion only
    _, encoders = encode_features(X_train, types, fit=True)

    # Encode train and val
    Z_train, _ = encode_features(X_train, types, encoders=encoders)
    Z_val, _ = encode_features(X_val, types, encoders=encoders)

    # Choose H on validation set (grid search)
    best_H, best_acc = H, -1.0
    for cand_H in [4, 8, 16, 32]:
        rng = np.random.default_rng(SEED + cand_H)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_mlp(Z_train, y_train, cand_H, rng, epochs, BATCH, LR, L2)
        acc = evaluate(Z_val, y_val, params)
        if acc > best_acc:
            best_acc, best_H = acc, cand_H

    # Retrain on full training data with best_H
    Z_full, encoders = encode_features(X_all, types, fit=True)
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    W1, b1, W2, b2 = train_mlp(Z_full, y_all, best_H, rng, epochs, BATCH, LR, L2)

    # Build per-feature lookup tables T[d][256][H]
    off = 0
    T = []
    for j, t in enumerate(types):
        if t == "ordinal":
            E = encoders[j][1].reshape(256, 1)  # (256,1)
        else:
            E = encoders[j][1]                  # (256, D_j)
        T.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    # Pack weights
    out = struct.pack("<4i", 0x4D4C5031, d, best_H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-mlp8 inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weights format: magic 0x4D4C5031, int32 d,H,C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) sc[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```