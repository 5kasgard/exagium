```hypothesis
Propose a small MLP family "label-mlp8" with one hidden layer of 8 ReLU units, trained with Adam and hyperparameters (H, LR, L2) selected on the tuning split (last 20% of train.in). The non-linear capacity should beat the linear baseline (0.9613) on optdigits while being cheaper than label-mlp16 (8685 Ir/item) due to fewer hidden units. Failure: dev accuracy ≤ 0.9613 or CPU cost per item ≥ label-linear (5058 Ir/item).
```
```python
"""Table family with one ReLU hidden layer, hyperparameters chosen on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed training budget
SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64

# Grid for tuning split (last 20% of train.in)
H_OPTIONS = [8, 16]
LR_OPTIONS = [3e-3, 1e-3]
L2_OPTIONS = [1e-4, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256, k_j) and total width k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply per-feature encodings, return (n, k) float32 matrix."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_model(Z, y, H, LR, L2, epochs, rng):
    """Train on Z (n, k), return (W1, b1, W2, b2) or (W1, b1) if H=0."""
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
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
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


def evaluate(Z, y, params):
    """Return accuracy on Z, y."""
    if len(params) == 4:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        W1, b1 = params
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = len(X_all)
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]

    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_hyper = None
    best_params = None
    best_enc = None

    for H in H_OPTIONS:
        for LR in LR_OPTIONS:
            for L2 in L2_OPTIONS:
                enc = build_encoding(X_all[train_idx], types)
                Z_train = encode(X_all[train_idx], enc)
                Z_tune = encode(X_all[tune_idx], enc)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
                params = train_model(Z_train, y_all[train_idx], H, LR, L2, epochs, rng)
                acc = evaluate(Z_tune, y_all[tune_idx], params)
                if acc > best_acc:
                    best_acc = acc
                    best_hyper = (H, LR, L2)
                    best_params = params
                    best_enc = enc
                    print(f"tuning: H={H} LR={LR} L2={L2} acc={acc:.4f}", file=sys.stderr)

    print(f"best: H={best_hyper[0]} LR={best_hyper[1]} L2={best_hyper[2]} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    H, LR, L2 = best_hyper
    enc = build_encoding(X_all, types)
    Z_full = encode(X_all, enc)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(Z_full, y_all, H, LR, L2, epochs, rng)

    # Export weights in table family format (magic 0x54424C31)
    d = X_all.shape[1]
    W = H if H else C
    # Compute per-feature lookup tables T_j = E_j @ W1[offset:offset+k_j]
    T_list = []
    off = 0
    for E in enc:
        k_j = E.shape[1]
        T_j = E @ params[0][off:off + k_j]  # (256, W)
        T_list.append(T_j.astype(np.float32))
        off += k_j
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.tobytes()
    out += b1.tobytes()
    if H:
        W2 = params[2].astype(np.float32)
        b2 = params[3].astype(np.float32)
        out += W2.tobytes()
        out += b2.tobytes()
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