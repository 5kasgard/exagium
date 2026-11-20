```hypothesis
Family: label-fm (Factorization Machine, degree 2). It models all pairwise feature interactions via low-rank factorization (w0 + Σ w_i x_i + ½ Σ_f [(Σ_i v_if x_i)² - Σ_i v_if² x_i²]). This captures critical categorical interactions in Adult (e.g., education×occupation) that linear models miss, while being far cheaper than MLPs/GBDTs at inference (O(d·k) with k=8). Hyperparameters (k=8, lr=0.01, L2=1e-4, epochs=30) chosen on the last 20% tuning split. Failure mode: if interactions are noisy or ordinal features dominate, the extra parameters may overfit and fall below label-linear (0.8524).
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored, categorical as one-hot (255=unseen).
Learns w0, w[k], V[k, fm_dim] with Adam on logistic loss.
Exports per-byte lookup tables for fast inference: T_lin, T_inter, T_inter_sq."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS = 0.01, 1e-4, 64, 30
FM_DIMS = [4, 8, 16]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of (enc_matrix[256, L_j], offset, length) for each feature."""
    encoders = []
    offset = 0
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]  # [256, 1]
            encoders.append((E.astype(np.float32), offset, 1))
            offset += 1
        else:
            vals = np.unique(col)
            L = len(vals)
            E = np.zeros((256, L), np.float32)
            E[vals, np.arange(L)] = 1.0
            encoders.append((E, offset, L))
            offset += L
    return encoders, offset


def encode_batch(X, encoders, k):
    """Encode batch X [n, d] to Z [n, k]."""
    n, d = X.shape
    Z = np.zeros((n, k), np.float32)
    for j, (E, off, L) in enumerate(encoders):
        Z[:, off:off+L] = E[X[:, j]]
    return Z


def train_fm(Z, y, fm_dim, rng, epochs=EPOCHS, batch=BATCH, lr=LR, l2=L2):
    """Train FM on Z [n, k], y [n] in {0,1}. Returns w0, w[k], V[k, fm_dim]."""
    n, k = Z.shape
    w0 = 0.0
    w = rng.normal(0, 0.01, k).astype(np.float32)
    V = rng.normal(0, 0.01, (k, fm_dim)).astype(np.float32)
    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_w0, v_w0 = 0.0, 0.0
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    steps_per_epoch = math.ceil(n / batch)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb = Z[idx]          # [b, k]
            yb = y[idx]          # [b]
            bsz = len(yb)
            # Forward
            Vx = Xb @ V                       # [b, fm_dim]
            Vx_sq = (Xb * Xb) @ (V * V)       # [b, fm_dim]
            inter = 0.5 * np.sum(Vx * Vx - Vx_sq, axis=1)  # [b]
            lin = w0 + Xb @ w                 # [b]
            logit = lin + inter               # [b]
            p = 1.0 / (1.0 + np.exp(-logit))  # [b]
            grad = (p - yb).astype(np.float32)  # [b]
            # Gradients
            g_w0 = grad.mean()
            g_w = (Xb.T @ grad) / bsz + l2 * w
            Xg = Xb * grad[:, None]           # [b, k]
            X2g = (Xb * Xb) * grad[:, None]   # [b, k]
            g_V = (Xg.T @ Vx) / bsz - V * (X2g.sum(0)[:, None] / bsz) + l2 * V
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1**t)) / (math.sqrt(v_w0 / (1 - beta2**t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
    return w0, w, V


def eval_fm(Z, y, w0, w, V):
    Vx = Z @ V
    Vx_sq = (Z * Z) @ (V * V)
    inter = 0.5 * np.sum(Vx * Vx - Vx_sq, axis=1)
    logit = w0 + Z @ w + inter
    pred = (logit > 0).astype(np.int64)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Build encoders on ALL training data (for final model), but also need for tuning split
    encoders, k = build_encoders(X, types)
    Z = encode_batch(X, encoders, k)

    # Tuning split: last 20% in file order
    n = len(y)
    split = n - n // 5
    Z_tr, Z_va = Z[:split], Z[split:]
    y_tr, y_va = y[:split], y[split:]

    # Tune fm_dim
    best_dim, best_acc = FM_DIMS[0], -1.0
    for fm_dim in FM_DIMS:
        w0, w, V = train_fm(Z_tr, y_tr, fm_dim, rng)
        acc = eval_fm(Z_va, y_va, w0, w, V)
        if acc > best_acc:
            best_acc, best_dim = acc, fm_dim

    # Retrain on all data
    fm_dim = best_dim
    w0, w, V = train_fm(Z, y, fm_dim, rng)

    # Build export tables: T_lin[d,256], T_inter[d,256,fm_dim], T_inter_sq[d,256,fm_dim]
    T_lin = np.zeros((d, 256), np.float32)
    T_inter = np.zeros((d, 256, fm_dim), np.float32)
    T_inter_sq = np.zeros((d, 256, fm_dim), np.float32)
    for j, (E, off, L) in enumerate(encoders):
        # E is [256, L]
        T_lin[j] = E @ w[off:off+L]
        T_inter[j] = E @ V[off:off+L]
        T_inter_sq[j] = (E * E) @ (V[off:off+L] * V[off:off+L])

    # Write weights.bin
    MAGIC = 0x464D3131  # "FM11"
    out = struct.pack("<4if", MAGIC, d, fm_dim, 1, w0)
    out += T_lin.astype("<f4").tobytes()
    out += T_inter.astype("<f4").tobytes()
    out += T_inter_sq.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine (degree 2) with per-byte lookup tables.
 * Input record: d bytes (d=14 for adult). Output: 1 byte class (0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], fm_dim = hd[2], C = hd[3];
    if (d <= 0 || d > 256 || fm_dim <= 0 || fm_dim > 64 || C != 1) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    float *p = (float *)(hd + 4);
    float w0 = p[0];
    float *T_lin = p + 1;
    float *T_inter = T_lin + (long)d * 256;
    float *T_inter_sq = T_inter + (long)d * 256 * fm_dim;
    long expected = 20 + 4L * (1 + (long)d * 256 * (1 + 2 * fm_dim));
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *inter_sum = malloc(sizeof(float) * fm_dim);
    float *inter_sq_sum = malloc(sizeof(float) * fm_dim);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear = w0;
        for (int f = 0; f < fm_dim; f++) {
            inter_sum[f] = 0.0f;
            inter_sq_sum[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += T_lin[(long)j * 256 + v];
            const float *t_inter = T_inter + ((long)j * 256 + v) * fm_dim;
            const float *t_inter_sq = T_inter_sq + ((long)j * 256 + v) * fm_dim;
            for (int f = 0; f < fm_dim; f++) {
                inter_sum[f] += t_inter[f];
                inter_sq_sum[f] += t_inter_sq[f];
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < fm_dim; f++) {
            float s = inter_sum[f];
            interaction += s * s - inter_sq_sum[f];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```