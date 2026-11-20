```hypothesis
Family: label-fm (Factorization Machine on raw byte features)
Why: Adult has mixed ordinal/categorical features where pairwise interactions (e.g., education × occupation) matter.
     A degree-2 FM with small latent dimension k=8 captures these in O(d*k) time, beating linear while staying cheaper than GBDT.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ label-gbdt (7570).
```

```python
"""Factorization Machine for tabular byte features: train.py
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x464D3132, d, k, C, bias, w[d][256], V[d][256][k] (float32 little-endian)."""
import json, math, struct, sys

import numpy as np

# hyper-parameters (tuned on last 20% of train.in)
K_CHOICES = [4, 8, 16]
SEED = 1
LR = 0.01
L2 = 1e-4
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, bias, w, V, k):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    # linear part
    lin = bias + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part: 0.5*(||sum V||^2 - sum ||V||^2)
    # V shape (d, 256, k)
    S = np.zeros((n, k), dtype=np.float32)
    sq_norm = np.zeros(n, dtype=np.float32)
    for j in range(d):
        Vj = V[j, X[:, j], :]          # (n, k)
        S += Vj
        sq_norm += np.sum(Vj * Vj, axis=1)
    inter = 0.5 * (np.sum(S * S, axis=1) - sq_norm)
    return lin + inter


def train_fm(X_tr, y_tr, X_val, y_val, k):
    n, d = X_tr.shape
    rng = np.random.default_rng(SEED)
    # init
    bias = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, math.sqrt(2.0 / (256 * k)), size=(d, 256, k))).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    best_val = -1.0
    best_params = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_tr[idx]
            yb = y_tr[idx]
            bsz = len(yb)
            # forward
            lin = bias + np.sum(w[np.arange(d), xb.T], axis=0)
            S = np.zeros((bsz, k), dtype=np.float32)
            sq_norm = np.zeros(bsz, dtype=np.float32)
            for j in range(d):
                Vj = V[j, xb[:, j], :]
                S += Vj
                sq_norm += np.sum(Vj * Vj, axis=1)
            inter = 0.5 * (np.sum(S * S, axis=1) - sq_norm)
            score = lin + inter
            p = sigmoid(score)
            # gradient
            diff = p - yb  # (bsz,)
            # bias
            g_bias = diff.mean()
            # w and V
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                # linear gradient
                np.add.at(g_w[j], vals, diff)
                # interaction gradient
                Vj = V[j, vals, :]          # (bsz, k)
                # d_inter/dV = S - Vj
                dV = (S - Vj) * diff[:, None]
                np.add.at(g_V[j], vals, dV)
            # L2
            g_w += L2 * w
            g_V += L2 * V
            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9**t)) / (math.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * g_w
            v_w = 0.999 * v_w + 0.001 * g_w * g_w
            w -= LR * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
            # V
            m_V = 0.9 * m_V + 0.1 * g_V
            v_V = 0.999 * v_V + 0.001 * g_V * g_V
            V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)
        # validation
        val_scores = fm_forward(X_val, bias, w, V, k)
        val_pred = (val_scores > 0).astype(np.int64)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_params = (bias, w.copy(), V.copy())
    return best_params, best_val


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM only supports binary for now"

    # split 80/20 for tuning
    n = len(y)
    n_tr = int(0.8 * n)
    idx = np.arange(n)
    X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
    X_val, y_val = X[idx[n_tr:]], y[idx[n_tr:]]

    best_k = 8
    best_val = -1.0
    best_params = None
    for k in K_CHOICES:
        params, val_acc = train_fm(X_tr, y_tr, X_val, y_val, k)
        if val_acc > best_val:
            best_val = val_acc
            best_k = k
            best_params = params

    # retrain on full data with best_k
    bias, w, V = best_params
    # continue training on full data (few epochs)
    # We'll just do a few more epochs on full data
    n_full = n
    rng = np.random.default_rng(SEED)
    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)
            lin = bias + np.sum(w[np.arange(d), xb.T], axis=0)
            S = np.zeros((bsz, best_k), dtype=np.float32)
            sq_norm = np.zeros(bsz, dtype=np.float32)
            for j in range(d):
                Vj = V[j, xb[:, j], :]
                S += Vj
                sq_norm += np.sum(Vj * Vj, axis=1)
            inter = 0.5 * (np.sum(S * S, axis=1) - sq_norm)
            score = lin + inter
            p = sigmoid(score)
            diff = p - yb
            g_bias = diff.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, diff)
                Vj = V[j, vals, :]
                dV = (S - Vj) * diff[:, None]
                np.add.at(g_V[j], vals, dV)
            g_w += L2 * w
            g_V += L2 * V
            t += 1
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9**t)) / (math.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)
            m_w = 0.9 * m_w + 0.1 * g_w
            v_w = 0.999 * v_w + 0.001 * g_w * g_w
            w -= LR * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
            m_V = 0.9 * m_V + 0.1 * g_V
            v_V = 0.999 * v_V + 0.001 * g_V * g_V
            V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)

    # write weights.bin
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D3132, d, best_k, C)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class (0/1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || k < 1 || k > 64) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *bias = (float *)(hd + 4);
    float *w = bias + 1;
    float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = calloc(k, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = *bias;
        memset(S, 0, k * sizeof(float));
        float sq_norm = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += w[(long)j * 256 + v];
            float *Vj = V + ((long)j * 256 + v) * k;
            for (int l = 0; l < k; l++) {
                float val = Vj[l];
                S[l] += val;
                sq_norm += val * val;
            }
        }
        float inter = 0.0f;
        for (int l = 0; l < k; l++) inter += S[l] * S[l];
        inter = 0.5f * (inter - sq_norm);
        score += inter;
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(S);
    return 0;
}
```