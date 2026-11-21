```hypothesis
Family: label-elm (Extreme Learning Machine with random fixed first layer and ridge regression on output).
Why: ELM provides non-linear decision boundaries with minimal training cost (only solving a linear system). The fixed random projection expands the 64-dimensional input into a higher-dimensional space where handwritten digit classes become more linearly separable. On optdigits, a small ELM (H=20–50) can achieve >97% accuracy, beating logistic regression (96.1%) while inference cost remains low (similar to a small MLP).
Failure indicator: If dev accuracy is ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >2x, the hypothesis is falsified.
```

```python
"""label-elm: Extreme Learning Machine with random fixed first layer and ridge regression on output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seeds, single thread."""
import json, math, struct, sys

import numpy as np

MASTER_SEED = 12345

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_encoding(X, types):
    """Return list of encoding matrices E_j (256, k_j) for each byte position."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def apply_encoding(X, enc):
    """Apply encoding matrices to raw byte data X (n, d) -> Z (n, k)."""
    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(Z_parts, axis=1).astype(np.float32)

def relu(x):
    return np.maximum(x, 0)

def train_elm(Z, y, H, lam, seed):
    """Train ELM on encoded data Z (n, k), labels y (n,) int64.
    Returns W1 (k, H), b1 (H), W2 (H, C), b2 (C)."""
    n, k = Z.shape
    C = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    H_hidden = relu(Z @ W1 + b1)
    H_bias = np.hstack([H_hidden, np.ones((n, 1), dtype=np.float32)])
    Y = np.zeros((n, C), np.float32)
    Y[np.arange(n), y] = 1.0
    A = H_bias.T @ H_bias
    A.flat[::H+2] += lam
    B = H_bias.T @ Y
    W2_bias = np.linalg.solve(A, B)
    W2 = W2_bias[:H].astype(np.float32)
    b2 = W2_bias[H].astype(np.float32)
    return W1, b1, W2, b2

def accuracy(Z, y, W1, b1, W2, b2):
    H_hidden = relu(Z @ W1 + b1)
    scores = H_hidden @ W2 + b2
    pred = scores.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Split: last 20% for tuning (file order)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train_raw, y_train = X[:split], y[:split]
    X_tune_raw, y_tune = X[split:], y[split:]

    # Encoding on training split only
    enc_train = compute_encoding(X_train_raw, types)
    Z_train = apply_encoding(X_train_raw, enc_train)
    Z_tune = apply_encoding(X_tune_raw, enc_train)

    # Hyperparameter grid
    H_grid = [10, 20, 30, 40, 50]
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    best_acc = -1.0
    best_H = H_grid[0]
    best_lam = lam_grid[0]
    best_W1 = best_b1 = best_W2 = best_b2 = None

    for H in H_grid:
        for lam in lam_grid:
            seed = MASTER_SEED + H * 1000 + int(lam * 10000)
            W1, b1, W2, b2 = train_elm(Z_train, y_train, H, lam, seed)
            acc = accuracy(Z_tune, y_tune, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc = acc
                best_H = H
                best_lam = lam
                best_W1, best_b1, best_W2, best_b2 = W1, b1, W2, b2

    # Retrain on all data with best hyperparameters
    enc_full = compute_encoding(X, types)
    Z_full = apply_encoding(X, enc_full)
    seed = MASTER_SEED + best_H * 1000 + int(best_lam * 10000)
    W1, b1, W2, b2 = train_elm(Z_full, y, best_H, best_lam, seed)

    # Build lookup tables T for each original byte position
    T = []
    off = 0
    for j, E in enumerate(enc_full):
        Tj = (E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32)
        T.append(Tj)
        off += E.shape[1]

    # Write weights.bin in label-linear format
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += best_b1.astype("<f4").tobytes()
    out += best_W2.astype("<f4").tobytes()
    out += best_b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format: magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H>0: W2[H][C], b2[C] (W=H). */
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