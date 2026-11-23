```hypothesis
Family: label-mlp-tuned – a one-hidden-layer MLP (ReLU) whose hidden size H is chosen from {8,16,32,64} on the tuning split (last 20% of train.in).  
Why: Optdigits has non-linear decision boundaries; a small MLP adds capacity with minimal inference overhead (H=8 costs ~same as linear). Tuning H on the validation split adapts model complexity to the data.  
Failure: If the best validation H gives accuracy ≤ label-linear (0.9613) or inference cost exceeds label-linear significantly, the family does not improve.
```

```python
"""label-mlp-tuned: one-hidden-layer MLP with ReLU, hidden size tuned on validation split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy+stdlib only.
Encodes ordinal features by z-score (on training split), categorical by one-hot.
Trains with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports weights in the same table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = [8, 16, 32, 64]  # hidden sizes to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit_idx):
    """Compute encoding matrices from fit_idx rows of X, return list of (256, dim_j) float64."""
    enc = []
    for j, t in enumerate(types):
        v = X[fit_idx, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals))); E[vals.astype(int), np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoding(X, enc):
    """Apply precomputed encoding matrices to X, return float32 matrix (n, total_dim)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - one_hot) / n."""
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)


def train_model(Z, y, H, C, rng):
    """Train on Z (n, k) and y (n,), return params: W1, b1, (W2, b2 if H>0)."""
    n, k = Z.shape
    W = H if H else C
    # Xavier init
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def eval_accuracy(Z, y, params, H, C):
    """Forward pass, return accuracy."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def write_weights(enc, params, H, C, d, out_path):
    """Fold encodings into first-layer tables and write binary weights."""
    W1, b1 = params[0], params[1]
    W = H if H else C
    off, T = 0, []
    for E in enc:
        # E is (256, dim_j) float64 -> float32
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    open(out_path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = len(X)
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    # Compute encodings on training split only
    enc_train = encode_features(X, types, train_idx)
    Z_train = apply_encoding(X[train_idx], enc_train)
    Z_val = apply_encoding(X[val_idx], enc_train)
    y_train, y_val = y[train_idx], y[val_idx]

    best_H, best_acc = None, -1.0
    for H in H_CANDIDATES:
        params = train_model(Z_train, y_train, H, C, rng)
        acc = eval_accuracy(Z_val, y_val, params, H, C)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on full data with best_H
    enc_full = encode_features(X, types, np.arange(n))
    Z_full = apply_encoding(X, enc_full)
    params = train_model(Z_full, y, best_H, C, rng)
    write_weights(enc_full, params, best_H, C, d, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* label-mlp-tuned inference: ./infer weights.bin in.rec N
 * Same format as label-linear (magic 0x54424C31, d, H, C, then tables). */
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