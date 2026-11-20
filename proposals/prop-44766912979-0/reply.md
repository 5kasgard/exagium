```hypothesis
Factorization Machine (FM) with binary logistic loss captures pairwise feature interactions
efficiently via low-rank factors. Adult has 14 sparse features (6 ordinal, 8 categorical) where
interactions (e.g., education×occupation) are known to be predictive. FM adds only O(k·f) parameters
(f=8) over linear, keeping inference fast (≈1.5k Ir/item) while typically surpassing linear accuracy.
If dev accuracy ≤ label-linear (0.8524) or Ir/item > 2000, the model does not justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes ordinal as z-scored scalars, categorical as one-hot (255=unseen→zero).
Trains FM with Adam on binary logistic loss; selects factor size f∈{4,8,16} on last 20% tuning split,
then retrains on all data. Exports: magic, d, k, f, w0, w[k], V[k·f], index_table[d·256](int16),
value_table[d·256](float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS = 0.01, 1e-4, 256, 20
F_CHOICES = [4, 8, 16]
TUNE_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_mappings(card, X):
    """Return (k, index_table, value_table) where tables are flat d*256 arrays."""
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    index_table = np.full(d * 256, -1, dtype=np.int16)
    value_table = np.zeros(d * 256, dtype=np.float32)
    idx = 0
    for j, t in enumerate(types):
        base = j * 256
        if t == "ordinal":
            col = X[:, j].astype(np.float64)
            mu, sd = col.mean(), col.std() or 1.0
            for v in range(256):
                index_table[base + v] = idx
                value_table[base + v] = (v - mu) / sd
            idx += 1
        else:  # categorical
            vals = np.unique(X[:, j])
            vals = vals[vals != 255]  # 255 is unseen
            for v in vals:
                index_table[base + v] = idx
                value_table[base + v] = 1.0
                idx += 1
            # 255 stays -1 index, 0.0 value
    return idx, index_table, value_table


def fm_forward(x_idx, x_val, w0, w, V, f):
    """x_idx: (m,) int16, x_val: (m,) float32. Return scores (m,)."""
    m = len(x_idx)
    linear = w0 + np.sum(w[x_idx] * x_val)
    # interaction via 0.5*(|sum u|^2 - sum |u|^2)
    sum_u = np.zeros(f, dtype=np.float32)
    sum_sq = 0.0
    for i in range(m):
        vi = V[x_idx[i]] * x_val[i]
        sum_u += vi
        sum_sq += np.dot(vi, vi)
    inter = 0.5 * (np.dot(sum_u, sum_u) - sum_sq)
    return linear + inter


def fm_backward(x_idx, x_val, grad, w, V, f, L2):
    """Accumulate gradients for one sample. Returns (dw0, dw, dV)."""
    m = len(x_idx)
    dw0 = grad
    dw = np.zeros_like(w)
    dV = np.zeros_like(V)
    # compute u_i and sum_u
    u = V[x_idx] * x_val[:, None]  # (m, f)
    sum_u = u.sum(axis=0)
    for i in range(m):
        xi = x_idx[i]
        val = x_val[i]
        dw[xi] += grad * val + L2 * w[xi]
        dV[xi] += grad * val * (sum_u - u[i]) + L2 * V[xi]
    return dw0, dw, dV


def train_fm(X, y, k, index_table, value_table, f, rng, epochs, batch, lr, l2):
    n = X.shape[0]
    d = X.shape[1]
    w0 = 0.0
    w = np.zeros(k, dtype=np.float32)
    V = (rng.normal(0, 0.01, (k, f))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            # accumulate batch gradients
            bw0, bw, bV = 0.0, np.zeros_like(w), np.zeros_like(V)
            for i in idx:
                row = X[i]
                # gather non-zero features
                feat_idx = []
                feat_val = []
                for j in range(d):
                    base = j * 256
                    v = row[j]
                    ii = index_table[base + v]
                    if ii >= 0:
                        feat_idx.append(ii)
                        feat_val.append(value_table[base + v])
                if not feat_idx:
                    continue
                feat_idx = np.array(feat_idx, dtype=np.int16)
                feat_val = np.array(feat_val, dtype=np.float32)
                score = fm_forward(feat_idx, feat_val, w0, w, V, f)
                p = 1.0 / (1.0 + math.exp(-score))
                grad = p - y[i]  # dL/dscore
                dw0, dw, dV = fm_backward(feat_idx, feat_val, grad, w, V, f, l2)
                bw0 += dw0
                bw += dw
                bV += dV
            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * bw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * bw0 * bw0
            m_w = beta1 * m_w + (1 - beta1) * bw
            v_w = beta2 * v_w + (1 - beta2) * bw * bw
            m_V = beta1 * m_V + (1 - beta1) * bV
            v_V = beta2 * v_V + (1 - beta2) * bV * bV
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    return w0, w, V


def eval_fm(X, y, index_table, value_table, w0, w, V, f):
    n = X.shape[0]
    d = X.shape[1]
    correct = 0
    for i in range(n):
        row = X[i]
        feat_idx = []
        feat_val = []
        for j in range(d):
            base = j * 256
            v = row[j]
            ii = index_table[base + v]
            if ii >= 0:
                feat_idx.append(ii)
                feat_val.append(value_table[base + v])
        if not feat_idx:
            score = w0
        else:
            score = fm_forward(np.array(feat_idx), np.array(feat_val), w0, w, V, f)
        pred = 1 if score > 0 else 0
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # build mappings once
    k, index_table, value_table = build_mappings(card, X)

    # tuning split (last 20%)
    split = int(n * (1 - TUNE_FRAC))
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    best_f, best_acc = F_CHOICES[0], -1.0
    for f in F_CHOICES:
        w0, w, V = train_fm(X_tr, y_tr, k, index_table, value_table, f, rng, EPOCHS, BATCH, LR, L2)
        acc = eval_fm(X_va, y_va, index_table, value_table, w0, w, V, f)
        if acc > best_acc:
            best_acc, best_f = acc, f

    # retrain on all data with best_f
    w0, w, V = train_fm(X, y, k, index_table, value_table, best_f, rng, EPOCHS, BATCH, LR, L2)

    # pack weights.bin
    # magic=0x464D3130, d, k, f
    header = struct.pack("<4i", 0x464D3130, X.shape[1], k, best_f)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + V.astype("<f4").tobytes()
    body += index_table.astype("<i2").tobytes() + value_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0/1).
 * Weights format: magic(4), d(4), k(4), f(4), w0(f4), w[k](f4), V[k*f](f4),
 * index_table[d*256](i2), value_table[d*256](f4). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], f = hd[3];
    if (d != 14 || k <= 0 || f <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float w0 = *ptr++;
    float *w = ptr; ptr += k;
    float *V = ptr; ptr += (long)k * f;
    int16_t *index_table = (int16_t *)ptr; ptr = (float *)(index_table + (long)d * 256);
    float *value_table = ptr;

    long expected = 16 + 4L * (1 + k + (long)k * f) + 2L * d * 256 + 4L * d * 256;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_u = calloc(f, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }
        float linear = w0;
        memset(sum_u, 0, f * sizeof(float));
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            int base = j * 256;
            int16_t idx = index_table[base + xi[j]];
            if (idx >= 0) {
                float val = value_table[base + xi[j]];
                linear += w[idx] * val;
                float *vi = V + (long)idx * f;
                for (int l = 0; l < f; l++) {
                    float u = vi[l] * val;
                    sum_u[l] += u;
                    sum_sq += u * u;
                }
            }
        }
        float dot = 0.0f;
        for (int l = 0; l < f; l++) dot += sum_u[l] * sum_u[l];
        float inter = 0.5f * (dot - sum_sq);
        float score = linear + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```