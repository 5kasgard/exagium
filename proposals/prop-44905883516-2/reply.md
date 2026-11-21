```hypothesis
Family: label-embmlp (embedding MLP with one hidden layer). Each of the 16 context bytes gets a learned embedding; their sum passes through a ReLU layer (size tuned from {32,64,128} on the last 20% of training data) then a linear layer to 65 classes. This captures pairwise interactions between positions that the linear model misses, while inference stays fast (sum of 16 embedding vectors + one ReLU layer). If dev accuracy does not exceed label-linear's 0.3375, the added capacity does not help for this context length.
```

```python
"""Embedding MLP family for next-byte prediction: one hidden ReLU layer on summed per-position embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on last 20% of train.in)
H_CHOICES = [32, 64, 128]
SEED = 1
LR = 3e-3
L2 = 1e-4
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


def build_encodings(X, types):
    """Return list of (E_j, num_unique_j) for each position j."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append((E, 1))
        else:  # categorical
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append((E, len(vals)))
    return enc


def encode_dataset(X, enc):
    """Concatenate one-hot/ordinal encodings for all positions."""
    parts = []
    for j, (E, _) in enumerate(enc):
        parts.append(E[X[:, j]])
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_mlp(Z, y, H, C, epochs, batch, lr, l2, rng):
    """Train one-hidden-layer MLP with Adam. Returns W1, b1, W2, b2."""
    n, k = Z.shape
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
            gh = g @ W2.T
            gh[a <= 0] = 0.0
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = len(X)
    split = int(0.8 * n)
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    rng_base = np.random.default_rng(SEED)

    # Hyperparameter search on tuning split
    best_H = None
    best_acc = -1.0
    for H in H_CHOICES:
        rng = np.random.default_rng(rng_base.integers(2**32))
        enc = build_encodings(X[train_idx], types)
        Z_train = encode_dataset(X[train_idx], enc)
        Z_tune = encode_dataset(X[tune_idx], enc)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
        params = train_mlp(Z_train, y[train_idx], H, C, epochs, BATCH, LR, L2, rng)
        acc = evaluate(Z_tune, y[tune_idx], params)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Final training on all data
    rng = np.random.default_rng(rng_base.integers(2**32))
    enc = build_encodings(X, types)
    Z = encode_dataset(X, enc)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W1, b1, W2, b2 = train_mlp(Z, y, best_H, C, epochs, BATCH, LR, L2, rng)

    # Build per-position lookup tables T_j = E_j @ W1_slice
    d = X.shape[1]
    W = best_H
    off = 0
    T_list = []
    for j, (E, dim) in enumerate(enc):
        T_list.append((E @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # shape (d, 256, W)

    # Write weights.bin: magic, d, H, C; T (d*256*W float32); b1 (W); W2 (H*C); b2 (C)
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding MLP (one hidden ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H). */
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