```hypothesis
We propose a tuned small MLP family (label-mlp-tune) that selects the hidden layer size H ∈ {4,8,12,16} on the last 20% of the training data (by file order), then retrains on the full training set. The optdigits pixels are ordinal and likely benefit from modest non‑linear interactions; a linear model reaches 96.13% accuracy, while a fixed 16‑unit MLP reaches 97.70% at higher cost. By tuning H we expect to exceed label‑linear accuracy (≥96.5%) while keeping inference cost well below label‑mlp16 (≈6000–7000 Ir/item). If the best tuned H does not beat label‑linear on dev, or if its inference cost exceeds label‑mlp16, the family fails its goal.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_H = [4, 8, 12, 16]  # hidden sizes to try on validation split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def apply_encoders(X, enc):
    """Transform X (n x d) to Z (n x k) using precomputed encoders."""
    n = X.shape[0]
    parts = []
    for j, E in enumerate(enc):
        parts.append(E[X[:, j]])
    return np.concatenate(parts, axis=1).astype(np.float32)

def train_model(Z, y, H, epochs, rng):
    """Train on Z (n x k), return params [W1, b1, (W2, b2)]."""
    n, k = Z.shape
    C = int(y.max()) + 1
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
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def predict(Z, params, H):
    """Return predicted class indices for Z."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        W1, b1 = params
        scores = Z @ W1 + b1
    return scores.argmax(axis=1)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Validation split: last 20% in file order
    n_val = n // 5
    n_train = n - n_val
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n)

    rng = np.random.default_rng(SEED)

    # Try each H on validation split
    best_H = None
    best_acc = -1.0
    for H in CANDIDATE_H:
        # Encoders from training split only
        enc = build_encoders(X[train_idx], types)
        Z_train = apply_encoders(X[train_idx], enc)
        Z_val = apply_encoders(X[val_idx], enc)

        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        params = train_model(Z_train, y[train_idx], H, epochs, rng)
        pred = predict(Z_val, params, H)
        acc = (pred == y[val_idx]).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on full data with best_H
    enc_full = build_encoders(X, types)
    Z_full = apply_encoders(X, enc_full)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params_final = train_model(Z_full, y, best_H, epochs_full, rng)

    # Build lookup tables T for export (same format as label-linear/mlp16)
    off = 0
    W = best_H if best_H else C
    T_list = []
    for E in enc_full:
        dim = E.shape[1]
        T_list.append((E.astype(np.float32) @ params_final[0][off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # shape (d, 256, W)

    # Pack weights.bin
    # magic, d, H, C
    out = struct.pack("<4i", 0x54424C31, X.shape[1], best_H, C)
    out += T.astype("<f4").tobytes()
    out += params_final[1].astype("<f4").tobytes()  # b1
    if best_H:
        out += params_final[2].astype("<f4").tobytes()  # W2
        out += params_final[3].astype("<f4").tobytes()  # b2

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