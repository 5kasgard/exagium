```hypothesis
We propose label-mlp-opt: a one-hidden-layer MLP with per-byte feature encoding (same as label-linear) and a ReLU hidden layer of tunable size (16, 32, 64). Hyperparameters (hidden size, learning rate, L2) are selected on the last 20% tuning split, then the model is retrained on all training data. This captures non-linear interactions missed by the linear model while keeping inference cost moderate via precomputed lookup tables. If dev accuracy does not exceed label-linear's 0.8524, or if inference cost exceeds label-mlp16's ~2300 Ir/item, the family does not justify its complexity.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed seed for overall determinism
SEED = 1
# Grid search space
H_CHOICES = [16, 32, 64]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
MAX_EPOCHS_GRID = 50
MAX_EPOCHS_FULL = 100
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encodings(X, types):
    """Return list of encoding matrices (256 x V_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode_batch(X, enc):
    """X: (n, d) uint8 -> (n, k) float32"""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_mlp(Z, y, H, C, lr, l2, epochs, rng):
    n, k = Z.shape
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
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + l2 * W1, gh.sum(0), h.T @ g + l2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + l2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def eval_accuracy(Z, y, params, H, C):
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

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Build encodings on training split only
    enc = build_encodings(X_train, types)
    Z_train = encode_batch(X_train, enc)
    Z_val = encode_batch(X_val, enc)

    # Grid search
    best_acc = -1
    best_params = None
    best_H = None
    best_lr = None
    best_l2 = None
    rng_base = np.random.default_rng(SEED)
    for H in H_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                rng = np.random.default_rng(rng_base.integers(2**32))
                params = train_mlp(Z_train, y_train, H, C, lr, l2, MAX_EPOCHS_GRID, rng)
                acc = eval_accuracy(Z_val, y_val, params, H, C)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_H = H
                    best_lr = lr
                    best_l2 = l2

    # Retrain on full data with best hyperparameters
    enc_full = build_encodings(X, types)
    Z_full = encode_batch(X, enc_full)
    rng_full = np.random.default_rng(SEED)
    final_params = train_mlp(Z_full, y, best_H, C, best_lr, best_l2, MAX_EPOCHS_FULL, rng_full)

    # Prepare lookup tables T for inference: d x 256 x H
    if best_H:
        W1, b1, W2, b2 = final_params
    else:
        W1, b1 = final_params
        W2 = None
        b2 = None
    k = W1.shape[1] if best_H else W1.shape[1]  # W1 is (k, H) or (k, C)
    # W1 is (k, H) or (k, C). We need to split W1 per feature.
    # enc_full[j] is (256, V_j). Contribution of feature j to first layer: enc_full[j] @ W1[offset:offset+V_j]
    T_list = []
    off = 0
    for j, E in enumerate(enc_full):
        Vj = E.shape[1]
        Tj = E @ W1[off:off+Vj]  # (256, H) or (256, C)
        T_list.append(Tj.astype(np.float32))
        off += Vj
    T = np.stack(T_list)  # (d, 256, H) or (d, 256, C)

    # Write weights.bin
    # Magic: 0x4D4C5031 ("MLP1")
    H_out = best_H
    C_out = C
    W = H_out if H_out else C_out
    # Check sizes
    magic = 0x4D4C5031
    header = struct.pack("<4i", magic, d, H_out, C_out)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H_out:
        body += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
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
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *s = malloc(sizeof(float) * C);
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