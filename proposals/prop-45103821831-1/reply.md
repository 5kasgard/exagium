```hypothesis
Family: label-mlp8 (single ReLU layer with 8 hidden units). Optdigits 8x8 pixels are low-dimensional and ordinal; a tiny MLP adds nonlinearity with minimal parameters (64*256*8 + 8 + 8*10 + 10 ≈ 130k weights) and should outperform linear (0.961) while staying cheaper than mlp16 (8685 Ir/item). If dev accuracy ≤ 0.961 or Ir/item ≥ 9000, the model fails its goal.
```

```python
"""Table family with one ReLU layer (H=8) and hyperparameter tuning on the last 20% of train.in.
python3 -s -P train.py card.json train.in train.key weights.bin   (deterministic, single thread)
Encoding: per-pixel z-scoring (ordinal). Adam lr/L2 grid search on validation split, then retrain on all data.
Exported as per-byte lookup tables: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H:
float32 W2[H][C], b2[C] (W = H)."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8
SEED = 1
# Hyperparameter grid for tuning
LR_CANDIDATES = [1e-3, 3e-3, 1e-2]
L2_CANDIDATES = [1e-4, 1e-3]
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


def build_enc(X):
    """X: (n, d) uint8. Returns list of (256, 1) float64 encoding matrices for each pixel."""
    d = X.shape[1]
    enc = []
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        enc.append(((np.arange(256, dtype=np.float64) - mu) / sd)[:, None])
    return enc


def encode(X, enc):
    """X: (n, d) uint8. Returns Z: (n, d) float32."""
    d = X.shape[1]
    Z = np.empty((X.shape[0], d), dtype=np.float32)
    for j in range(d):
        Z[:, j] = enc[j][X[:, j], 0]
    return Z


def softmax_grad(logits, y):
    """logits: (batch, C), y: (batch,) int64. Returns gradient w.r.t logits."""
    S = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)


def train_model(Z, y, C, H, lr, L2, batch, max_epochs, seed):
    """Train on Z (n, k), return params and Adam state."""
    n, k = Z.shape
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
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def accuracy(Z, y, params, H, C):
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
    n = X.shape[0]
    val_size = n // 5
    # Split: first 80% train, last 20% validation (tuning split)
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    # Build encoding on training part only for tuning
    enc_train = build_enc(X_train)
    Z_train = encode(X_train, enc_train)
    Z_val = encode(X_val, enc_train)

    k = Z_train.shape[1]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_acc = -1.0
    best_params = None
    best_lr = best_l2 = None

    for i, lr in enumerate(LR_CANDIDATES):
        for j, L2 in enumerate(L2_CANDIDATES):
            seed = SEED + i * len(L2_CANDIDATES) + j
            params = train_model(Z_train, y_train, C, H, lr, L2, BATCH, max_epochs, seed)
            acc = accuracy(Z_val, y_val, params, H, C)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_lr, best_l2 = lr, L2

    # Retrain on full data with best hyperparameters
    enc_full = build_enc(X)
    Z_full = encode(X, enc_full)
    max_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(Z_full, y, C, H, best_lr, best_l2, BATCH, max_epochs_full, SEED)

    # Build lookup tables T for inference
    W1, b1 = final_params[0], final_params[1]
    W = H if H else C
    off = 0
    T_list = []
    for E in enc_full:
        # E is (256,1), W1 slice is (1, W)
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)

    # Write weights.bin
    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2, b2 = final_params[2], final_params[3]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
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