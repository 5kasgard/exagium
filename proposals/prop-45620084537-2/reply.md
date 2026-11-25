```hypothesis
We propose a two-layer MLP with ReLU hidden layer (family "label-tinymlp"). The 16-byte context is one-hot encoded per position, summed into a hidden layer of size H, then a linear output layer. This adds nonlinear capacity while keeping inference fast (16*H + H*C operations). Tuning H on the validation split should yield better accuracy than linear (0.3375) with modest cost increase. If accuracy does not exceed 0.34 or cost exceeds 2x linear, the model fails.
```
```python
"""Two-layer MLP (ReLU hidden) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Encodes each byte position as one-hot (categorical). Searches H on last 20% of train,
then retrains on all data. Exports binary weights: magic 0x4D4C5032, d, H, C;
float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [16, 32, 64, 128]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_encoders(card, X):
    """Return list of encoding matrices E_j (256 x dim_j) for each position."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
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


def train_model(X, y, enc, H, epochs, rng):
    """Train MLP with given hidden size, return (T, b1, W2, b2)."""
    C = int(y.max()) + 1
    d = X.shape[1]
    # Build design matrix Z (n x sum dim_j)
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    n, k = Z.shape
    # Init weights
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # Backprop
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh *= (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Convert W1 to per-position lookup tables T[d][256][H]
    T = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    return np.stack(T), b1, W2, b2


def evaluate(X, y, enc, T, b1, W2, b2):
    """Compute accuracy on given data."""
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    a = Z @ T.reshape(-1, T.shape[-1]) + b1  # T is (d,256,H) -> reshape to (d*256, H)
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    # Split: last 20% for tuning
    split = int(0.8 * n_all)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    enc = make_encoders(card, X_all)
    # Hyperparameter search on validation set
    best_H, best_acc = H_CHOICES[0], -1.0
    base_rng = np.random.default_rng(SEED)
    for H in H_CHOICES:
        rng = np.random.default_rng(base_rng.integers(2**32))
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        T, b1, W2, b2 = train_model(X_train, y_train, enc, H, epochs, rng)
        acc = evaluate(X_val, y_val, enc, T, b1, W2, b2)
        if acc > best_acc:
            best_acc, best_H = acc, H
    # Retrain on all data with best_H
    rng = np.random.default_rng(base_rng.integers(2**32))
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    T, b1, W2, b2 = train_model(X_all, y_all, enc, best_H, epochs, rng)
    # Write weights
    d, H, C = X_all.shape[1], best_H, int(y_all.max()) + 1
    magic = 0x4D4C5032  # 'MLP2'
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Two-layer MLP (ReLU hidden) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (categorical). Output: one byte class index.
 * Weights format: magic 0x4D4C5032, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
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
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) sc[c] += hv * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```