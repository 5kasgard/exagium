```hypothesis
We propose a tiny one-hidden-layer MLP (label-tinymlp) with 8 ReLU units. The optdigits pixels have non-linear interactions that a linear model (label-linear, 0.9613) cannot capture, while a 16-unit MLP (label-mlp16, 0.9770) shows the potential. An 8-unit MLP should keep most of the accuracy gain with lower inference cost. If dev accuracy ≤ 0.9613 or instructions/item exceed label-mlp16, the family fails its goal.
```

```python
"""label-tinymlp: small MLP with one hidden layer for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exports weights in the same Table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
H_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
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


def prepare_encodings(X, types):
    """Return list of encoding matrices (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                               # (256, n_cat)
    return enc


def encode_data(X, enc):
    """Transform raw byte matrix X (n x d) to concatenated encoding Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_model(X_tr, y_tr, X_val, y_val, H, lr, seed):
    """Train on (X_tr, y_tr), return (val_acc, params) where params = (W1,b1,W2,b2,enc)."""
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    C = card["classes"]
    enc = prepare_encodings(X_tr, types)
    Z_tr = encode_data(X_tr, enc)
    Z_val = encode_data(X_val, enc)
    n, k = Z_tr.shape

    rng = np.random.default_rng(seed)
    # He initialization for ReLU
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            gh = g @ W2.T
            gh[a <= 0] = 0.0
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    a_val = Z_val @ W1 + b1
    h_val = np.maximum(a_val, 0)
    logits_val = h_val @ W2 + b2
    preds = logits_val.argmax(axis=1)
    val_acc = (preds == y_val).mean()
    return val_acc, (W1, b1, W2, b2, enc)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning (in file order)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_cfg = None
    best_params = None
    cfg_idx = 0
    for H in H_CHOICES:
        for lr in LR_CHOICES:
            # Deterministic seed per configuration
            seed = SEED + cfg_idx
            cfg_idx += 1
            val_acc, params = train_model(X_tr, y_tr, X_val, y_val, H, lr, seed)
            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg = (H, lr)
                best_params = params

    # Retrain on full training set with best config
    H, lr = best_cfg
    _, (W1, b1, W2, b2, enc) = train_model(X, y, X[:1], y[:1], H, lr, SEED)  # dummy val
    # The above returns params trained on full X (since we passed full X as train and tiny val).
    # But train_model uses the passed training set for encodings and training. We need encodings from full X.
    # Let's recompute encodings on full X and retrain properly.
    enc_full = prepare_encodings(X, card["input"]["types"])
    Z_full = encode_data(X, enc_full)
    n_full, k_full = Z_full.shape
    rng = np.random.default_rng(SEED)
    W1 = (rng.uniform(-1, 1, (k_full, H)) * math.sqrt(6.0 / k_full)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_full[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            gh = g @ W2.T
            gh[a <= 0] = 0.0
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables T for each feature: T[j] = enc_full[j] @ W1_slice
    d = X.shape[1]
    W = H
    off = 0
    T_list = []
    for j, E in enumerate(enc_full):
        dim_j = E.shape[1]
        T_list.append((E.astype(np.float32) @ W1[off:off + dim_j]).astype(np.float32))
        off += dim_j
    T = np.stack(T_list)  # (d, 256, W)

    # Write weights.bin
    magic = 0x54424C31
    header = struct.pack("<4i", magic, d, H, C)
    out = header + T.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
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