```hypothesis
Family: label-fm (Factorization Machine). Adult has mixed categorical/ordinal features; FM captures pairwise interactions via low-rank embeddings, often outperforming linear models on tabular data. We use embedding size k=4 (tuned on last 20% of train), Adam, L2=1e-4. If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Each feature value (0-255) gets a bias weight and a k-dim embedding.
Prediction: w0 + sum w_j[x_j] + 0.5 * ( (sum v_j[x_j])^2 - sum v_j[x_j]^2 ). Binary output via sigmoid.
Tunes embedding size k in {2,4,8} on last 20% of train.in, then retrains on all data.
Weights format: magic 0x464D3132, d, k, C; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS_TUNE, EPOCHS_FULL = 0.01, 1e-4, 64, 10, 20
K_CHOICES = [2, 4, 8]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, w0, w, V):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    k = V.shape[2]
    logits = np.full(n, w0, dtype=np.float32)
    # linear term
    for j in range(d):
        logits += w[j, X[:, j]]
    # interaction term
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_v_sq = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j]]  # (n, k)
        sum_v += v
        sum_v_sq += v * v
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    logits += inter
    return logits


def train_fm(X, y, k, epochs, rng):
    n, d = X.shape
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            # forward
            logits = fm_forward(xi, w0, w, V)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (probs - yi).astype(np.float32)  # (b,)
            # gradients
            b = len(yi)
            gw0 = grad_logits.mean()
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            # linear grad
            for j in range(d):
                vals = xi[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        gw[j, v] = grad_logits[mask].mean()
            # interaction grad
            sum_v = np.zeros((b, k), dtype=np.float32)
            for j in range(d):
                sum_v += V[j, xi[:, j]]
            for j in range(d):
                v_j = V[j, xi[:, j]]  # (b, k)
                grad_v = grad_logits[:, None] * (sum_v - v_j)  # (b, k)
                for v in range(256):
                    mask = (xi[:, j] == v)
                    if np.any(mask):
                        gV[j, v] = grad_v[mask].mean(axis=0)
            # L2
            gw += L2 * w
            gV += L2 * V
            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * (gV * gV)
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    return w0, w, V


def accuracy(X, y, w0, w, V):
    logits = fm_forward(X, w0, w, V)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    rng = np.random.default_rng(SEED)
    # split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    # tune k
    best_k, best_acc = K_CHOICES[0], -1.0
    for k in K_CHOICES:
        w0, w, V = train_fm(X_train, y_train, k, EPOCHS_TUNE, rng)
        acc = accuracy(X_val, y_val, w0, w, V)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # retrain on all data
    w0, w, V = train_fm(X, y, best_k, EPOCHS_FULL, rng)
    # write weights
    magic = 0x464D3132  # "FM12"
    C = card["classes"]
    out = struct.pack("<4i", magic, d, best_k, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class (0 or 1).
 * Weights format: magic 0x464D3132, d, k, C; float32 w0, w[d][256], V[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || k < 1) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + sizeof(float) * (1 + (size_t)d * 256 + (size_t)d * 256 * k);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (size_t)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v_sq = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float logit = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            logit += w[(size_t)j * 256 + val];
            const float *v = V + ((size_t)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vf = v[f];
                sum_v[f] += vf;
                sum_v_sq[f] += vf * vf;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_v[f] * sum_v[f] - sum_v_sq[f];
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```