```hypothesis
Family: label-fm (Factorization Machine on the label-linear encoding).
Why: Adult has strong pairwise interactions (e.g., education × occupation) that linear models miss; FM adds a low-rank interaction term (k=8) over the same 14-byte input, keeping inference at ~14·k·C FLOPs. The encoding (ordinal z-scored, categorical one-hot) yields exactly 14 non-zeros per row, so the FM forward pass is a tiny dense loop.
Failure signal: dev accuracy ≤ label-linear (0.8524) or instructions/item > 2× label-linear.
```

```python
"""Factorization Machine on the label-linear encoding.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D4C31, d=14, k, C=2; float32 bias[C], T_lin[C][d][256], T_lat[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [4, 8, 16]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(card, X):
    """Return list of encoding matrices E_j (256 x dim_j) and slice boundaries."""
    enc, slices = [], []
    off = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # 256 x 1
            dim = 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0                # 256 x n_cat
            dim = len(vals)
        enc.append(E.astype(np.float32))
        slices.append((off, off + dim))
        off += dim
    return enc, slices, off


def encode_batch(X, enc, slices):
    """X: (n, d) uint8 -> Z: (n, d_enc) float32."""
    n, d = X.shape
    d_enc = slices[-1][1]
    Z = np.empty((n, d_enc), np.float32)
    for j, (E, (lo, hi)) in enumerate(zip(enc, slices)):
        Z[:, lo:hi] = E[X[:, j]]
    return Z


def fm_forward(Z, W, V, bias, k):
    """Z: (n, d_enc), W: (d_enc, C), V: (d_enc, k, C), bias: (C,) -> logits (n, C)."""
    n = Z.shape[0]
    C = bias.shape[0]
    lin = Z @ W + bias                                         # (n, C)
    inter = np.zeros((n, C), np.float32)
    for f in range(k):
        ZV = Z @ V[:, f, :]                                    # (n, C)
        inter += 0.5 * (ZV * ZV - (Z * Z) @ (V[:, f, :] * V[:, f, :]))
    return lin + inter


def softmax_grad(logits, y):
    """logits (n, C), y (n,) -> grad (n, C) of cross-entropy."""
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return P / len(y)


def train_fm(Z, y, k, rng, n_epochs):
    n, d_enc = Z.shape
    C = 2
    W = (rng.uniform(-1, 1, (d_enc, C)) * math.sqrt(6.0 / d_enc)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d_enc, k, C)) * math.sqrt(6.0 / (d_enc * k))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [W, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = fm_forward(z, W, V, bias, k)
            g = softmax_grad(logits, yy)                       # (B, C)
            # gradients
            gW = z.T @ g + L2 * W
            gbias = g.sum(0)
            gV = np.zeros_like(V)
            for f in range(k):
                ZV = z @ V[:, f, :]
                gZV = g * ZV                                   # (B, C)
                gV[:, f, :] = z.T @ gZV - (z * z).T @ (g * V[:, f, :]) + L2 * V[:, f, :]
            grads = [gW, gV, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, V, bias


def evaluate(Z, y, W, V, bias, k):
    logits = fm_forward(Z, W, V, bias, k)
    pred = logits.argmax(1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2

    # encoding on full training data (used for final model)
    enc_full, slices_full, d_enc_full = build_encoding(card, X)
    Z_full = encode_batch(X, enc_full, slices_full)

    # validation split (last 20% in file order)
    n = len(X)
    split = int(0.8 * n)
    idx_train, idx_val = np.arange(split), np.arange(split, n)
    Z_tr, y_tr = Z_full[idx_train], y[idx_train]
    Z_val, y_val = Z_full[idx_val], y[idx_val]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_tr)))
    for k in K_CANDIDATES:
        W, V, bias = train_fm(Z_tr, y_tr, k, np.random.default_rng(SEED), n_epochs)
        acc = evaluate(Z_val, y_val, W, V, bias, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # final training on all data with best_k
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, V, bias = train_fm(Z_full, y, best_k, np.random.default_rng(SEED), n_epochs)

    # build lookup tables T_lin[C][d][256], T_lat[C][d][256][k]
    T_lin = np.zeros((C, d, 256), np.float32)
    T_lat = np.zeros((C, d, 256, best_k), np.float32)
    for c in range(C):
        for j, (E, (lo, hi)) in enumerate(zip(enc_full, slices_full)):
            W_block = W[lo:hi, c]
            V_block = V[lo:hi, :, c]
            T_lin[c, j] = E @ W_block
            T_lat[c, j] = E @ V_block

    # write weights.bin
    magic = 0x464D4C31  # 'FML1'
    hdr = struct.pack("<4i", magic, d, best_k, C)
    out = hdr + bias.astype("<f4").tobytes() + T_lin.astype("<f4").tobytes() + T_lat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class index 0/1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 32) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * 256 + 4L * C * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *T_lin = bias + C;
    const float *T_lat = T_lin + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if (rec_len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        for (int c = 0; c < C; c++) {
            float lin = bias[c];
            for (int j = 0; j < d; j++) lin += T_lin[(c * d + j) * 256 + xi[j]];
            float inter = 0.0f;
            for (int f = 0; f < k; f++) {
                float sum_vx = 0.0f, sum_vx2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float t = T_lat[((c * d + j) * 256 + xi[j]) * k + f];
                    sum_vx += t;
                    sum_vx2 += t * t;
                }
                inter += 0.5f * (sum_vx * sum_vx - sum_vx2);
            }
            logits[c] = lin + inter;
        }
        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```