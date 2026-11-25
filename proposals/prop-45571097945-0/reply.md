```hypothesis
Family: label-fm (Factorization Machine). FMs model all pairwise feature interactions via low-rank latent vectors, which is ideal for tabular data where combinations (e.g., education×occupation) matter. With latent dim k=4–8 and precomputed lookup tables, inference stays O(d·k) ~ 14·8 = 112 ops/item, far below mlp16's cost. Expect dev accuracy > 0.855 at < 1500 Ir/item. Failure if accuracy ≤ 0.852 or Ir/item > 2000.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal -> z-scored (1 dim), categorical -> one-hot over training values.
FM: w0 + w·x + 0.5 Σ_f [(Σ_i V_if x_i)² - Σ_i V_if² x_i²] per class.
Tuned on last 20% of train (latent dim k∈{2,4,8,16}, LR∈{1e-3,3e-3}), then retrained on all.
Exported as lookup tables: magic 0x464D4C31, d=14, k, C, D; float32 w0[C], L[14][256][C], VF[14][256][k][C], VF2[14][256][k][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4
K_CHOICES = [2, 4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(card, X):
    """Return (encoders, offsets, D) where encoders[j] maps uint8->float32 vector of len D_j."""
    encoders = []
    offsets = []
    D = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu = v.mean()
            sd = v.std() or 1.0
            # 256 values -> (256, 1)
            E = ((np.arange(256, dtype=np.float32) - mu) / sd).reshape(256, 1)
            encoders.append(E)
            offsets.append(D)
            D += 1
        else:
            vals = np.unique(v)
            nvals = len(vals)
            E = np.zeros((256, nvals), dtype=np.float32)
            E[vals, np.arange(nvals)] = 1.0
            encoders.append(E)
            offsets.append(D)
            D += nvals
    return encoders, offsets, D


def encode_batch(X, encoders, offsets, D):
    """X: (n, 14) uint8 -> Z: (n, D) float32"""
    n = X.shape[0]
    Z = np.zeros((n, D), dtype=np.float32)
    for j in range(14):
        E = encoders[j]
        off = offsets[j]
        d = E.shape[1]
        Z[:, off:off+d] = E[X[:, j]]
    return Z


def fm_forward(Z, w0, w, V, k, C):
    """Z: (n, D), w0: (C), w: (D, C), V: (D, k, C) -> scores (n, C)"""
    n = Z.shape[0]
    lin = Z @ w + w0  # (n, C)
    if k == 0:
        return lin
    inter = np.zeros((n, C), dtype=np.float32)
    for f in range(k):
        Vf = V[:, f, :]  # (D, C)
        sum_vx = Z @ Vf  # (n, C)
        sum_v2x2 = (Z * Z) @ (Vf * Vf)  # (n, C)
        inter += 0.5 * (sum_vx * sum_vx - sum_v2x2)
    return lin + inter


def fm_backward(Z, y, scores, w0, w, V, k, C, L2):
    """Return gradients (dw0, dw, dV) for cross-entropy loss."""
    n = Z.shape[0]
    # softmax
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # (n, C)
    dw0 = probs.sum(axis=0)  # (C)
    dw = Z.T @ probs + L2 * w  # (D, C)
    dV = np.zeros_like(V)  # (D, k, C)
    if k > 0:
        for f in range(k):
            Vf = V[:, f, :]  # (D, C)
            sum_vx = Z @ Vf  # (n, C)
            # d/dV_{i,f,c} = x_i * (sum_j V_{j,f,c} x_j - V_{i,f,c} x_i)
            # For each sample: x_i * (sum_vx[n,c] - Vf[i,c] * x_i)
            # Vectorized: (Z[:,:,None] * (sum_vx[:,None,:] - Vf[None,:,:] * Z[:,:,None])).sum(axis=0)
            # Z: (n,D), sum_vx: (n,C), Vf: (D,C)
            # We want (D,C): sum_n Z[n,i] * (sum_vx[n,c] - Vf[i,c] * Z[n,i])
            term = sum_vx[:, None, :] - Vf[None, :, :] * Z[:, :, None]  # (n, D, C)
            grad = (Z[:, :, None] * term).sum(axis=0)  # (D, C)
            dV[:, f, :] = grad + L2 * Vf
    return dw0, dw, dV


def train_fm(Z, y, k, C, lr, epochs, batch, rng):
    n, D = Z.shape
    w0 = np.zeros(C, np.float32)
    w = np.zeros((D, C), np.float32)
    V = (rng.normal(0, 0.01, (D, k, C))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = np.zeros(C, np.float32), np.zeros(C, np.float32)
    m_w, v_w = np.zeros((D, C), np.float32), np.zeros((D, C), np.float32)
    m_V, v_V = np.zeros((D, k, C), np.float32), np.zeros((D, k, C), np.float32)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Zb = Z[idx]
            yb = y[idx]
            scores = fm_forward(Zb, w0, w, V, k, C)
            dw0, dw, dV = fm_backward(Zb, yb, scores, w0, w, V, k, C, L2)
            t += 1
            # Adam update
            for param, grad, m, v in [(w0, dw0, m_w0, v_w0),
                                       (w, dw, m_w, v_w),
                                       (V, dV, m_V, v_V)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w0, w, V


def eval_acc(Z, y, w0, w, V, k, C):
    scores = fm_forward(Z, w0, w, V, k, C)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def build_tables(encoders, offsets, D, w0, w, V, k, C):
    """Precompute lookup tables for all 14*256 raw values."""
    # linear_table[j][v][c] = sum_i w[off+i, c] * E[v][i]
    linear_table = np.zeros((14, 256, C), dtype=np.float32)
    vf_table = np.zeros((14, 256, k, C), dtype=np.float32)
    vf2_table = np.zeros((14, 256, k, C), dtype=np.float32)
    for j in range(14):
        E = encoders[j]  # (256, d_j)
        off = offsets[j]
        d = E.shape[1]
        w_block = w[off:off+d]  # (d_j, C)
        V_block = V[off:off+d]  # (d_j, k, C)
        for v in range(256):
            ev = E[v]  # (d_j)
            # linear
            linear_table[j, v] = ev @ w_block
            if k > 0:
                for f in range(k):
                    Vf = V_block[:, f, :]  # (d_j, C)
                    vf_table[j, v, f] = ev @ Vf
                    vf2_table[j, v, f] = (ev * ev) @ (Vf * Vf)
    return linear_table, vf_table, vf2_table


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 14)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(y)
    rng = np.random.default_rng(SEED)

    # Build encoding on full training data (for final model), but for tuning we need consistent encoding
    # We'll build encoding on full data, then split indices.
    encoders, offsets, D = build_encoding(card, X)
    Z = encode_batch(X, encoders, offsets, D)

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    Z_train, y_train = Z[idx_train], y[idx_train]
    Z_tune, y_tune = Z[idx_tune], y[idx_tune]

    best_acc = -1
    best_cfg = None
    best_params = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_train)))
            w0, w, V = train_fm(Z_train, y_train, k, C, lr, epochs, BATCH, rng)
            acc = eval_acc(Z_tune, y_tune, w0, w, V, k, C)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (k, lr, epochs)
                best_params = (w0.copy(), w.copy(), V.copy())
            print(f"k={k} lr={lr} tune_acc={acc:.4f}", file=sys.stderr)

    k, lr, epochs = best_cfg
    print(f"Best: k={k} lr={lr} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    w0, w, V = train_fm(Z, y, k, C, lr, epochs, BATCH, rng)

    # Build lookup tables
    linear_table, vf_table, vf2_table = build_tables(encoders, offsets, D, w0, w, V, k, C)

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<5i", magic, 14, k, C, D)
    out = header
    out += w0.astype("<f4").tobytes()
    out += linear_table.astype("<f4").tobytes()
    out += vf_table.astype("<f4").tobytes()
    out += vf2_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 raw bytes. Output: 1 byte class index (argmax). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    uint8_t *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3], D = hd[4];
    if (d != 14 || C < 1 || C > 256 || k < 0 || k > 32) { fprintf(stderr, "bad header\n"); return 3; }

    /* Layout after header (5 int32 = 20 bytes):
     * float32 w0[C]
     * float32 linear_table[14][256][C]
     * float32 vf_table[14][256][k][C]
     * float32 vf2_table[14][256][k][C]
     */
    size_t off = 20;
    size_t need = off
        + (size_t)C * 4
        + (size_t)14 * 256 * C * 4
        + (size_t)14 * 256 * k * C * 4
        + (size_t)14 * 256 * k * C * 4;
    if (nw != (long)need) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, need); return 3; }

    float *w0 = (float *)(wb + off); off += C * 4;
    float *lin = (float *)(wb + off); off += 14 * 256 * C * 4;
    float *vf  = (float *)(wb + off); off += 14 * 256 * k * C * 4;
    float *vf2 = (float *)(wb + off);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, len); return 3; }

        for (int c = 0; c < C; c++) scores[c] = w0[c];

        /* linear part */
        for (int j = 0; j < 14; j++) {
            const float *lj = lin + ((j * 256 + xi[j]) * C);
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        /* FM interactions */
        if (k > 0) {
            for (int f = 0; f < k; f++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f, sum2 = 0.0f;
                    for (int j = 0; j < 14; j++) {
                        size_t idx = ((j * 256 + xi[j]) * k + f) * C + c;
                        sum  += vf[idx];
                        sum2 += vf2[idx];
                    }
                    scores[c] += 0.5f * (sum * sum - sum2);
                }
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```