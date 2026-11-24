```hypothesis
Family: label-linear-cat – linear model with per-pixel one-hot encoding (treat ordinal 0..16 as categorical).
Why: Pixel intensities in optdigits have non-linear class responses; one-hot adds expressivity (17× more params per pixel) while keeping identical inference structure (per-byte lookup tables) and cost.
Failure signal: dev accuracy ≤ 0.9613 (label-linear baseline) or weights.bin size mismatch.
```
```python
"""label-linear-cat: linear classifier with categorical (one-hot) encoding for ordinal features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2_GRID = [1e-5, 1e-4, 1e-3, 1e-2]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Build per-feature one-hot encoding matrices (256 x n_vals) from data X."""
    enc = []
    for j, t in enumerate(types):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    return enc

def apply_encoding(X, enc):
    """Apply encoding to data X, return concatenated feature matrix (n x total_dim)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_linear(Z, y, C, L2, epochs, rng):
    """Train linear softmax with Adam (H=0). Returns W1 (k x C), b1 (C)."""
    n, k = Z.shape
    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W1 + b1, yy)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W1, b1

def evaluate_linear(W1, b1, Z, y):
    scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n_all = len(X_all)
    split = int(n_all * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    # Tune L2 on tuning split
    enc_train = build_encoding(X_train, types)
    Z_train = apply_encoding(X_train, enc_train)
    Z_tune = apply_encoding(X_tune, enc_train)

    best_L2 = L2_GRID[0]
    best_acc = -1.0
    for L2 in L2_GRID:
        rng = np.random.default_rng(SEED)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        W1, b1 = train_linear(Z_train, y_train, C, L2, epochs, rng)
        acc = evaluate_linear(W1, b1, Z_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_L2 = L2

    # Retrain on all data with best L2
    enc_all = build_encoding(X_all, types)
    Z_all = apply_encoding(X_all, enc_all)
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    W1, b1 = train_linear(Z_all, y_all, C, best_L2, epochs, rng)

    # Export as Table family format (magic, d, H, C, T[d][256][C], b1[C])
    d = card["input"]["bytes"]
    H = 0
    off = 0
    T_list = []
    for E in enc_all:
        T_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T_list).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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