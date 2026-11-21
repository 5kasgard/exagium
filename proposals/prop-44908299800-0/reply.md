```hypothesis
Family: label-fm (Factorization Machine with shared embeddings).
Why: Adult has strong pairwise interactions (e.g., education×occupation, capital-gain×capital-loss). A linear model misses these; a full MLP is overkill. An FM captures all pairwise interactions via low-rank embeddings (k=4) with only ~21k parameters, adding ~O(d·k) FLOPs per item. It should beat label-linear (0.8524) while staying well below label-mlp16's cost (2325 Ir/item).
Failure signal: If dev accuracy ≤ 0.8524 or Ir/item > 1500, the family does not justify its complexity.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte as a category (0-255). Learns per-class linear weights and shared k-dim embeddings.
Scores: score_c = bias_c + sum_j w[j,x_j,c] + 0.5*sum_f((sum_j v[j,x_j,f])^2 - sum_j v[j,x_j,f]^2).
Hyperparameters (lr, L2, k) chosen on last 20% of train.in (tuning split), then retrained on all train."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Hyperparameter grid for tuning split
GRID = {
    "lr": [1e-2, 3e-3, 1e-3],
    "l2": [1e-4, 1e-3],
    "k": [2, 4, 8],
}


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_tuning(X, y, frac=0.2):
    n = X.shape[0]
    cut = int(n * (1 - frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def init_params(d, C, k, rng):
    # bias [C], linear weights [d,256,C], embeddings [d,256,k]
    bias = np.zeros(C, np.float32)
    w = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    v = (rng.uniform(-1, 1, (d, 256, k)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    return bias, w, v


def forward(Xb, bias, w, v):
    """Xb: (B,d) uint8. Returns scores (B,C), and intermediates for backward."""
    B, d = Xb.shape
    C = bias.shape[0]
    k = v.shape[2]
    # linear part: bias + sum_j w[j, x_j, :]
    lin = np.zeros((B, C), np.float32)
    lin += bias
    for j in range(d):
        lin += w[j, Xb[:, j], :]
    # interaction part
    sum_v = np.zeros((B, k), np.float32)
    sum_v2 = np.zeros((B, k), np.float32)
    for j in range(d):
        vj = v[j, Xb[:, j], :]  # (B,k)
        sum_v += vj
        sum_v2 += vj * vj
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (B,)
    scores = lin + inter[:, None]
    return scores, sum_v, sum_v2


def backward(Xb, yb, scores, sum_v, sum_v2, bias, w, v, l2):
    """Compute gradients for a batch. Returns (gbias, gw, gv)."""
    B, d = Xb.shape
    C = bias.shape[0]
    k = v.shape[2]
    # softmax gradient
    P = np.exp(scores - scores.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(B), yb] -= 1.0
    P /= B  # (B,C)
    # linear gradients
    gbias = P.sum(axis=0).astype(np.float32)
    gw = np.zeros_like(w)
    for j in range(d):
        idx = Xb[:, j]
        for c in range(C):
            gw[j, idx, c] += P[:, c]
    # interaction gradients
    dinter = P.sum(axis=1)  # (B,)
    # d(inter)/d(v_{j,x_j,f}) = sum_v[f] - v_{j,x_j,f}
    gv = np.zeros_like(v)
    for j in range(d):
        idx = Xb[:, j]
        vj = v[j, idx, :]  # (B,k)
        # (sum_v - vj) * dinter[:,None]
        gv_j = (sum_v - vj) * dinter[:, None]
        for f in range(k):
            gv[j, idx, f] += gv_j[:, f]
    # L2 regularization
    gw += l2 * w
    gv += l2 * v
    return gbias, gw, gv


def train_epoch(X, y, bias, w, v, lr, l2, batch, rng, state, t):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch):
        idx = order[s:s + batch]
        Xb, yb = X[idx], y[idx]
        scores, sum_v, sum_v2 = forward(Xb, bias, w, v)
        gbias, gw, gv = backward(Xb, yb, scores, sum_v, sum_v2, bias, w, v, l2)
        t += 1
        # Adam update
        for param, grad, (m, v_) in zip([bias, w, v], [gbias, gw, gv], state):
            m *= 0.9; m += 0.1 * grad
            v_ *= 0.999; v_ += 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v_ / (1 - 0.999 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t


def accuracy(X, y, bias, w, v):
    scores, _, _ = forward(X, bias, w, v)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    # split tuning (last 20%)
    Xtr, ytr, Xval, yval = split_tuning(X, y, 0.2)
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for lr in GRID["lr"]:
        for l2 in GRID["l2"]:
            for k in GRID["k"]:
                bias, w, v = init_params(d, C, k, rng)
                state = [(np.zeros_like(bias), np.zeros_like(bias)),
                         (np.zeros_like(w), np.zeros_like(w)),
                         (np.zeros_like(v), np.zeros_like(v))]
                t = 0
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Xtr.shape[0]))
                for _ in range(epochs):
                    t = train_epoch(Xtr, ytr, bias, w, v, lr, l2, BATCH, rng, state, t)
                acc = accuracy(Xval, yval, bias, w, v)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (lr, l2, k)
                    best_params = (bias.copy(), w.copy(), v.copy())
    # retrain on full data with best config
    lr, l2, k = best_cfg
    bias, w, v = best_params
    state = [(np.zeros_like(bias), np.zeros_like(bias)),
             (np.zeros_like(w), np.zeros_like(w)),
             (np.zeros_like(v), np.zeros_like(v))]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))
    for _ in range(epochs):
        t = train_epoch(X, y, bias, w, v, lr, l2, BATCH, rng, state, t)
    # write weights: magic, d, k, C, bias[C], w[d,256,C], v[d,256,k]
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, k, C)
    out += bias.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (argmax of two scores). */
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
    if (C != 2 || d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *w = bias + C;
    const float *v = w + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16], sum_v2[16], score[2];
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        score[0] = bias[0]; score[1] = bias[1];
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *wj = w + ((long)j * 256 + val) * C;
            score[0] += wj[0]; score[1] += wj[1];
            const float *vj = v + ((long)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vf = vj[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        score[0] += inter; score[1] += inter;
        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```