```hypothesis
Family: label-mlp8 (table family with 8 ReLU units). The 8x8 digit pixels have strong local interactions that a linear model misses; a tiny hidden layer adds non-linearity with only 64*8 + 8*10 = 592 multiply-adds per item (cheaper than label-linear's 640 and far cheaper than label-mlp16's 1184). Hyperparameters (LR, L2, batch) are tuned on the last 20% of train.in. If dev accuracy ≤ 0.9613 (label-linear) or instruction cost ≥ label-mlp16, the family does not meet its goal.
```
```python
"""Table family with a small hidden layer (H=8) and hyperparameter tuning on the 20% tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
H = 8
SEED = 1
SAMPLES, MAX_EPOCHS = 600000, 100

# Hyperparameter grid for tuning
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-5, 1e-4, 1e-3]
BATCH_GRID = [32, 64, 128]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_encoders(X, types):
    """Return list of encoding matrices (256 x dim_j) computed on X."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoders(X, enc):
    """Apply encoders to X (n x d) -> Z (n x total_dim)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_eval(Z_tr, y_tr, Z_va, y_va, k, C, H, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, seed):
    """Train on Z_tr/y_tr, evaluate on Z_va/y_va. Return (accuracy, params)."""
    rng = np.random.default_rng(seed)
    n, _ = Z_tr.shape
    W = H if H else C
    # Initialize
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
            z, yy = Z_tr[idx], y_tr[idx]
            # Forward
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W1 + b1
            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # Backward
            if H:
                grad_W2 = h.T @ g + L2 * W2
                grad_b2 = g.sum(0)
                gh = (g @ W2.T) * (a > 0)
                grad_W1 = z.T @ gh + L2 * W1
                grad_b1 = gh.sum(0)
                grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W1 = z.T @ g + L2 * W1
                grad_b1 = g.sum(0)
                grads = [grad_W1, grad_b1]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Evaluate
    if H:
        a = Z_va @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z_va @ W1 + b1
    pred = logits.argmax(axis=1)
    acc = (pred == y_va).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = len(X)
    # Split: last 20% for tuning
    va_size = int(0.2 * n)
    tr_size = n - va_size
    X_tr, y_tr = X[:tr_size], y[:tr_size]
    X_va, y_va = X[tr_size:], y[tr_size:]

    # Build encoders on training split only
    enc = make_encoders(X_tr, types)
    Z_tr = apply_encoders(X_tr, enc)
    Z_va = apply_encoders(X_va, enc)
    k = Z_tr.shape[1]

    # Grid search
    best_acc, best_params, best_cfg = -1, None, None
    for i, LR in enumerate(LR_GRID):
        for j, L2 in enumerate(L2_GRID):
            for k_b, BATCH in enumerate(BATCH_GRID):
                seed = SEED + i * 100 + j * 10 + k_b
                acc, params = train_eval(Z_tr, y_tr, Z_va, y_va, k, C, H, LR, L2, BATCH,
                                         SAMPLES, MAX_EPOCHS, seed)
                if acc > best_acc:
                    best_acc, best_params, best_cfg = acc, params, (LR, L2, BATCH)
    print(f"Best tuning acc: {best_acc:.4f} with LR={best_cfg[0]}, L2={best_cfg[1]}, BATCH={best_cfg[2]}", file=sys.stderr)

    # Retrain on full data with best config
    enc_full = make_encoders(X, types)
    Z_full = apply_encoders(X, enc_full)
    _, final_params = train_eval(Z_full, y, Z_full, y, k, C, H, *best_cfg,
                                 SAMPLES, MAX_EPOCHS, SEED)

    # Export weights in table-family format (same as label-linear)
    W1, b1 = final_params[0], final_params[1]
    if H:
        W2, b2 = final_params[2], final_params[3]
    W = H if H else C
    off, T = 0, []
    for E in enc_full:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
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