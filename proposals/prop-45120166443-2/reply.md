```hypothesis
We propose "label-mlp-tuned": a one-hidden-layer MLP with ReLU, where hidden size, learning rate, and L2 are selected by grid search on the tuning split (last 20% of train). The model uses the same per-position one-hot encoding as label-linear but adds a hidden layer to capture non-linear interactions between context bytes. Since label-gbdt reaches 0.3923, a well-tuned MLP should bridge the gap from label-linear's 0.3375. If it fails to exceed 0.34, it would indicate that the task's non-linearities are not easily captured by a single hidden layer or that the hyperparameter space is insufficient.
```

```python
"""label-mlp-tuned: one-hidden-layer MLP with hyperparameter tuning on the tuning split.
train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

# Fixed constants
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid
H_OPTIONS = [64, 128, 256]
LR_OPTIONS = [1e-3, 3e-3]
L2_OPTIONS = [1e-4, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x d_j) and total input dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E.astype(np.float32))
    return enc


def train_model(X, y, enc, H, LR, L2, seed, epochs):
    """Train on given data, return parameters (W1, b1, W2, b2) and final loss."""
    rng = np.random.default_rng(seed)
    # Build Z matrix
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    n, k = Z.shape
    C = 65  # card["classes"]
    W = H if H else C

    # Initialize parameters
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(2.0 / k)).astype(np.float32)  # He init
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
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
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(X, y, enc, params, H):
    """Return accuracy on given data."""
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
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
    n = X.shape[0]
    tune_size = n // 5  # last 20%
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    enc = encode_features(X, card["input"]["types"])

    # Grid search
    best_acc = -1.0
    best_params = None
    best_hyper = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for H in H_OPTIONS:
        for LR in LR_OPTIONS:
            for L2 in L2_OPTIONS:
                # Use a distinct seed for each combination
                seed = 1000 + H * 100 + int(LR * 10000) + int(L2 * 10000)
                params = train_model(X_train, y_train, enc, H, LR, L2, seed, epochs)
                acc = evaluate(X_tune, y_tune, enc, params, H)
                print(f"H={H} LR={LR} L2={L2} tune_acc={acc:.4f}", file=sys.stderr)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_hyper = (H, LR, L2)

    print(f"Best: H={best_hyper[0]} LR={best_hyper[1]} L2={best_hyper[2]} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters and fixed seed
    H, LR, L2 = best_hyper
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, enc, H, LR, L2, SEED, epochs_full)

    # Build lookup tables T for each position
    W = H if H else 65
    off = 0
    T = []
    for E in enc:
        # E is 256 x d_j, final_params[0] is W1 (k x W)
        T.append((E @ final_params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T_stack = np.stack(T)  # d x 256 x W

    # Write weights.bin
    d = X.shape[1]
    C = 65
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += T_stack.astype("<f4").tobytes()
    out += final_params[1].astype("<f4").tobytes()  # b1
    if H:
        out += final_params[2].astype("<f4").tobytes()  # W2
        out += final_params[3].astype("<f4").tobytes()  # b2
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