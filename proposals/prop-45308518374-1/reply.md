```hypothesis
Family: label-mlp-tuned (one-hidden-layer MLP with per-byte embedding, hidden size and learning rate chosen on the 20% tuning split).
Why: optdigits has 64 ordinal pixel-count features (0..16) with non-linear structure; a small ReLU MLP captures interactions that linear models miss. The embedding-lookup first layer makes inference O(d·H) with d=64, H≤64, keeping CPU cost low. Tuning H∈{16,32,64} and LR∈{3e-3,1e-3} on the held-out 20% adapts to the task without overfitting.
Failure: dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-mlp16.
```

```python
"""label-mlp-tuned: one-hidden-layer MLP with per-byte embedding, hyperparameters tuned on the last 20% of train.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic, single thread, stdlib+numpy)"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED_BASE = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Grid search space
H_CHOICES = [16, 32, 64]
LR_CHOICES = [3e-3, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encodings(X, types):
    """Return list of encoding matrices E_j (256, k_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encodings(X, enc):
    """Transform X (n, d) to Z (n, sum k_j) by concatenating E_j[X[:,j]]."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_model(Z, y, H, LR, seed):
    """Train on Z (n, k), return params (W1, b1, W2, b2) with W2,b2=None if H=0."""
    n, k = Z.shape
    C = 10  # card["classes"] but fixed for this task
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
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
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


def accuracy(Z, y, params, H):
    """Compute accuracy on Z, y."""
    W1, b1 = params[0], params[1]
    if H:
        W2, b2 = params[2], params[3]
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    types = card["input"]["types"]

    # Split: last 20% for tuning (file order)
    n = len(X)
    n_train = int(0.8 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Grid search
    best_acc = -1.0
    best_cfg = None
    best_enc = None
    for i, H in enumerate(H_CHOICES):
        for j, LR in enumerate(LR_CHOICES):
            seed = SEED_BASE + 100 + i * len(LR_CHOICES) + j
            enc = compute_encodings(X_tr, types)
            Z_tr = apply_encodings(X_tr, enc)
            Z_val = apply_encodings(X_val, enc)
            params = train_model(Z_tr, y_tr, H, LR, seed)
            acc = accuracy(Z_val, y_val, params, H)
            if acc > best_acc or (acc == best_acc and H < best_cfg[0]):
                best_acc = acc
                best_cfg = (H, LR)
                best_enc = enc
    print(f"Selected H={best_cfg[0]}, LR={best_cfg[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    H, LR = best_cfg
    enc_full = compute_encodings(X, types)
    Z_full = apply_encodings(X, enc_full)
    params = train_model(Z_full, y, H, LR, SEED_BASE)

    # Build lookup tables T (d, 256, W) as in the example
    W = H if H else C
    W1, b1 = params[0], params[1]
    off, T = 0, []
    for E in enc_full:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T_stack = np.stack(T)  # (d, 256, W)

    # Write weights.bin with magic 0x4D4C5031 ("MLP1")
    MAGIC = 0x4D4C5031
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += T_stack.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp-tuned inference: ./infer weights.bin in.rec N
 * Weights format: magic 0x4D4C5031, int32 d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C].  W = H ? H : C.
 * Each input record is d bytes; output is one byte (class index). */
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
    const int32_t MAGIC = 0x4D4C5031;
    if (nw < 16 || hd[0] != MAGIC) { fprintf(stderr, "bad weights\n"); return 3; }
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