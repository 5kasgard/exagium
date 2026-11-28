```hypothesis
Factorization machines model pairwise feature interactions with O(d·K) parameters and inference cost, capturing non-linearities that are crucial for tabular data like Adult. A small factor dimension (K=4–8) should raise accuracy above label-linear while keeping CPU cost near the linear baseline. If dev accuracy does not exceed label-linear by at least 0.3 pp, the family does not justify its complexity.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Exports: magic 0x464D4C31, d, K, C; float32 linear_bias[C], linear_emb[d][256][C], factor_emb[d][256][K][C].
Training: Adam on 80% train, grid search K∈{4,8} and lr∈{1e-3,3e-3} on last 20% tuning split, then retrain on all data."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4
GRID_K = [4, 8]
GRID_LR = [1e-3, 3e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(Z, linear_bias, linear_emb, factor_emb, K, C):
    """Z: (n, d) uint8. Returns logits (n, C)."""
    n, d = Z.shape
    logits = np.broadcast_to(linear_bias, (n, C)).astype(np.float32)
    # linear part
    for j in range(d):
        logits += linear_emb[j, Z[:, j], :]
    # factorization part
    for c in range(C):
        sum_f = np.zeros((n, K), dtype=np.float32)
        sum_sq = np.zeros((n, K), dtype=np.float32)
        for j in range(d):
            v = factor_emb[j, Z[:, j], :, c]  # (n, K)
            sum_f += v
            sum_sq += v * v
        inter = 0.5 * np.sum(sum_f * sum_f - sum_sq, axis=1)
        logits[:, c] += inter
    return logits


def softmax_grad(logits, y):
    """logits (n, C), y (n,) -> grad (n, C)"""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_fm(X_tr, y_tr, X_va, y_va, d, C, K, lr, rng):
    n_tr = X_tr.shape[0]
    # init
    linear_bias = np.zeros(C, np.float32)
    linear_emb = (rng.uniform(-0.01, 0.01, (d, 256, C)) * math.sqrt(6.0 / 256)).astype(np.float32)
    factor_emb = (rng.uniform(-0.01, 0.01, (d, 256, K, C)) * math.sqrt(6.0 / 256)).astype(np.float32)
    params = [linear_bias, linear_emb, factor_emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_acc = -1.0
    best_params = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X_tr[idx], y_tr[idx]
            logits = fm_forward(z, linear_bias, linear_emb, factor_emb, K, C)
            g = softmax_grad(logits, yy)  # (B, C)
            # gradients
            g_bias = g.sum(axis=0).astype(np.float32)
            g_lin = np.zeros_like(linear_emb)
            g_fac = np.zeros_like(factor_emb)
            for j in range(d):
                xj = z[:, j]
                # linear emb grad
                for c in range(C):
                    g_lin[j, xj, c] += g[:, c]
                # factor emb grad
                for c in range(C):
                    sum_f = np.zeros((len(idx), K), dtype=np.float32)
                    for jj in range(d):
                        sum_f += factor_emb[jj, z[:, jj], :, c]
                    for k in range(K):
                        g_fac[j, xj, k, c] += (sum_f[:, k] - factor_emb[j, xj, k, c]) * g[:, c]
            # L2
            g_lin += L2 * linear_emb
            g_fac += L2 * factor_emb
            grads = [g_bias, g_lin, g_fac]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # validation accuracy
        va_logits = fm_forward(X_va, linear_bias, linear_emb, factor_emb, K, C)
        va_pred = va_logits.argmax(axis=1)
        acc = (va_pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
    return best_params, best_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_overall = None
    best_acc = -1.0
    best_K = None
    best_lr = None
    for K in GRID_K:
        for lr in GRID_LR:
            params, acc = train_fm(X_tr, y_tr, X_va, y_va, d, C, K, lr, rng)
            if acc > best_acc:
                best_acc = acc
                best_overall = params
                best_K = K
                best_lr = lr
    # retrain on all data with best hyperparams
    rng = np.random.default_rng(SEED)
    final_params, _ = train_fm(X, y, X[:1], y[:1], d, C, best_K, best_lr, rng)  # tiny val to reuse function
    linear_bias, linear_emb, factor_emb = final_params
    # pack weights
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, best_K, C)
    out += linear_bias.astype("<f4").tobytes()
    out += linear_emb.astype("<f4").tobytes()
    out += factor_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    const float *linear_bias = p; p += C;
    const float *linear_emb = p; p += (long)d * 256 * C;
    const float *factor_emb = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_f = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) logits[c] = linear_bias[c];
        for (int j = 0; j < d; j++) {
            const float *le = linear_emb + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += le[c];
        }
        for (int c = 0; c < C; c++) {
            memset(sum_f, 0, K * sizeof(float));
            memset(sum_sq, 0, K * sizeof(float));
            for (int j = 0; j < d; j++) {
                const float *fe = factor_emb + (((long)j * 256 + xi[j]) * K + 0) * C + c;
                for (int f = 0; f < K; f++) {
                    float v = fe[f * C];
                    sum_f[f] += v;
                    sum_sq[f] += v * v;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) inter += sum_f[f] * sum_f[f] - sum_sq[f];
            logits[c] += 0.5f * inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```