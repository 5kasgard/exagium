```hypothesis
Family: label-fm (Factorization Machine with low-rank embeddings).
Why: Adult has strong pairwise interactions (e.g., education × occupation, marital-status × relationship) that linear models miss. A second-order FM with a small embedding dimension (k=8) captures these with O(d·k) parameters and O(d·k) inference, staying far cheaper than GBDT/MLP while adding expressive power.
Failure signal: dev accuracy ≤ label-linear (0.8524) or Ir/item > 2× label-linear (≈1400).
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 w0[C], w[C][d][256], v[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, BATCH, EPOCHS_TUNE, EPOCHS_FULL = 0.01, 256, 5, 30
K_CHOICES = [4, 8, 16]          # embedding dimensions to try on tuning split
L2_W, L2_V = 1e-5, 1e-5         # regularization

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(x, w0, w, v, d, k, C):
    """x: (n, d) uint8. Returns logits (n, C)."""
    n = x.shape[0]
    logits = np.zeros((n, C), dtype=np.float32)
    flat_idx = x + np.arange(d, dtype=np.uint16) * 256          # (n, d)
    for c in range(C):
        logits[:, c] = w0[c]
        logits[:, c] += w[c].reshape(-1)[flat_idx].sum(axis=1)
        V = v[c].reshape(d * 256, k)[flat_idx]                  # (n, d, k)
        sum_v = V.sum(axis=1)                                   # (n, k)
        sum_v2 = (V * V).sum(axis=1)                            # (n, k)
        inter = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=1))
        logits[:, c] += inter
    return logits

def fm_backward(x, y, logits, w0, w, v, d, k, C, l2_w, l2_v):
    """Compute gradients for one batch. Returns dict of gradients."""
    n = x.shape[0]
    flat_idx = x + np.arange(d, dtype=np.uint16) * 256          # (n, d)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n                                                   # (n, C)

    gw0 = np.zeros(C, np.float32)
    gw = np.zeros((C, d, 256), np.float32)
    gv = np.zeros((C, d, 256, k), np.float32)

    for c in range(C):
        g = probs[:, c]                                          # (n,)
        gw0[c] = g.sum() + l2_w * w0[c]
        gw_flat = np.bincount(flat_idx.ravel(), weights=np.repeat(g, d),
                              minlength=d * 256).astype(np.float32)
        gw[c] = gw_flat.reshape(d, 256) + l2_w * w[c]

        V = v[c].reshape(d * 256, k)[flat_idx]                  # (n, d, k)
        sum_v = V.sum(axis=1)                                   # (n, k)
        # d/dV = (sum_v - V) * g[:, None, None]
        factor = (sum_v[:, None, :] - V) * g[:, None, None]     # (n, d, k)
        gv_flat = np.zeros((d * 256, k), np.float32)
        np.add.at(gv_flat, flat_idx.ravel(), factor.reshape(-1, k))
        gv[c] = gv_flat.reshape(d, 256, k) + l2_v * v[c]

    return gw0, gw, gv

def train_fm(X, y, d, k, C, epochs, rng):
    w0 = np.zeros(C, np.float32)
    w = np.zeros((C, d, 256), np.float32)
    v = rng.normal(0, 0.01, (C, d, 256, k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits = fm_forward(xb, w0, w, v, d, k, C)
            gw0, gw, gv = fm_backward(xb, yb, logits, w0, w, v, d, k, C, L2_W, L2_V)
            t += 1
            for arr, m, v_, grad in [(w0, m_w0, v_w0, gw0),
                                     (w, m_w, v_w, gw),
                                     (v, m_v, v_v, gv)]:
                m *= beta1; m += (1 - beta1) * grad
                v_ *= beta2; v_ += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v_ / (1 - beta2 ** t)
                arr -= LR * m_hat / (np.sqrt(v_hat) + eps)
    return w0, w, v

def accuracy(X, y, w0, w, v, d, k, C):
    logits = fm_forward(X, w0, w, v, d, k, C)
    return (logits.argmax(axis=1) == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CHOICES[0], -1.0
    for k in K_CHOICES:
        w0, w, v = train_fm(X_tr, y_tr, d, k, C, EPOCHS_TUNE, rng)
        acc = accuracy(X_va, y_va, w0, w, v, d, k, C)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # final training on all data
    w0, w, v = train_fm(X, y, d, best_k, C, EPOCHS_FULL, rng)

    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, best_k, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class index (argmax). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * 256 + 4L * C * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w = w0 + C;
    const float *v = w + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = w0[c];
            const float *wc = w + (long)c * d * 256;
            const float *vc = v + (long)c * d * 256 * k;
            for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

            for (int j = 0; j < d; j++) {
                unsigned char val = xi[j];
                int idx = j * 256 + val;
                logits[c] += wc[idx];
                const float *vj = vc + (long)idx * k;
                for (int f = 0; f < k; f++) {
                    float vjf = vj[f];
                    sum_v[f] += vjf;
                    sum_v2[f] += vjf * vjf;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < k; f++) inter += sum_v[f] * sum_v[f] - sum_v2[f];
            logits[c] += 0.5f * inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```