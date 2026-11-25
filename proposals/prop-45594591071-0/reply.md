```hypothesis
Family: label-fm (Factorization Machine with per-value embeddings).
Reason: Adult has strong pairwise interactions (e.g., education×occupation, marital×sex) that linear models miss.
A low-rank FM captures all pairwise interactions in O(d·K) time with d=14, K=8, adding only ~130 KB weights.
It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) while being faster than MLPs.
Failure: dev accuracy ≤ 0.8524 or inference cost > label-linear.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Uses last 20% of train for hyperparameter tuning.
Exports: int32 magic 0x464D4C31, d, K, C; float32 bias, then d×256×(1+K) table [w, emb...]."""
import json, math, struct, sys

import numpy as np

# Tuning grid (chosen on last 20% of train.in)
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.03, 0.1]
L2_CHOICES = [0.0, 1e-4]
EPOCHS_TUNE = 15
EPOCHS_FULL = 30
BATCH = 64
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X, bias, W, Emb, K):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    scores = np.full(n, bias, dtype=np.float32)
    sum_e = np.zeros((n, K), dtype=np.float32)
    sum_e_sq = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        v = X[:, j]
        scores += W[j, v]
        e = Emb[j, v]                 # (n, K)
        sum_e += e
        sum_e_sq += e * e
    interaction = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
    return scores + interaction

def train_fm(X_tr, y_tr, X_val, y_val, K, lr, l2, epochs, rng):
    n, d = X_tr.shape
    # init
    bias = float(np.log((y_tr.mean() + 1e-8) / (1 - y_tr.mean() + 1e-8)))
    W = np.zeros((d, 256), dtype=np.float32)
    Emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_b, v_b = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_E, v_E = np.zeros_like(Emb), np.zeros_like(Emb)
    t = 0
    best_val_acc = -1.0
    best_params = None
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            n_b = len(yb)
            scores = np.full(n_b, bias, dtype=np.float32)
            sum_e = np.zeros((n_b, K), dtype=np.float32)
            sum_e_sq = np.zeros((n_b, K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]
                scores += W[j, v]
                e = Emb[j, v]
                sum_e += e
                sum_e_sq += e * e
            interaction = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
            scores += interaction
            # loss gradient
            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)          # (n_b,)
            # gradients
            g_bias = grad.mean()
            g_W = np.zeros_like(W)
            g_E = np.zeros_like(Emb)
            for j in range(d):
                v = xb[:, j]
                # linear weight gradient
                for val in range(256):
                    mask = (v == val)
                    if np.any(mask):
                        g_W[j, val] = grad[mask].mean()
                # embedding gradient
                # d(interaction)/d e_{j,v,k} = (sum_e_k - e_{j,v,k}) * grad
                # vectorized over batch
                for val in range(256):
                    mask = (v == val)
                    if np.any(mask):
                        # sum_e for this batch and feature
                        se = sum_e[mask]                # (m, K)
                        ee = Emb[j, val]                # (K,)
                        g_E[j, val] = ((se - ee) * grad[mask][:, None]).mean(axis=0)
            # L2 on W and Emb (not bias)
            g_W += l2 * W
            g_E += l2 * Emb
            # Adam update
            t += 1
            # bias
            m_b = 0.9 * m_b + 0.1 * g_bias
            v_b = 0.999 * v_b + 0.001 * g_bias * g_bias
            bias -= lr * (m_b / (1 - 0.9**t)) / (math.sqrt(v_b / (1 - 0.999**t)) + 1e-8)
            # W
            m_W = 0.9 * m_W + 0.1 * g_W
            v_W = 0.999 * v_W + 0.001 * g_W * g_W
            W -= lr * (m_W / (1 - 0.9**t)) / (np.sqrt(v_W / (1 - 0.999**t)) + 1e-8)
            # Emb
            m_E = 0.9 * m_E + 0.1 * g_E
            v_E = 0.999 * v_E + 0.001 * g_E * g_E
            Emb -= lr * (m_E / (1 - 0.9**t)) / (np.sqrt(v_E / (1 - 0.999**t)) + 1e-8)
        # validation accuracy
        if X_val is not None:
            val_scores = fm_forward(X_val, bias, W, Emb, K)
            val_pred = (val_scores > 0).astype(np.int64)
            val_acc = (val_pred == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (bias, W.copy(), Emb.copy())
    if best_params is not None:
        bias, W, Emb = best_params
    return bias, W, Emb

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    # tuning
    best_acc = -1.0
    best_hp = None
    best_model = None
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                bias, W, Emb = train_fm(X[:split], y[:split], X[split:], y[split:], K, lr, l2, EPOCHS_TUNE, rng)
                val_scores = fm_forward(X[split:], bias, W, Emb, K)
                val_pred = (val_scores > 0).astype(np.int64)
                acc = (val_pred == y[split:]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (K, lr, l2)
                    best_model = (bias, W, Emb)
    # final training on all data
    K, lr, l2 = best_hp
    bias, W, Emb = train_fm(X, y, None, None, K, lr, l2, EPOCHS_FULL, rng)
    # write weights
    magic = 0x464D4C31  # 'FML1'
    C = 2
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    for j in range(d):
        for v in range(256):
            out += struct.pack("<f", W[j, v])
            out += Emb[j, v].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0/1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = *p++;
    const float *W = p; p += (long)d * 256;
    const float *Emb = p;               // d * 256 * K
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = (float *)malloc(K * sizeof(float));
    float *sum_e_sq = (float *)malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = bias;
        for (int kk = 0; kk < K; kk++) { sum_e[kk] = 0.0f; sum_e_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            score += W[(long)j * 256 + v];
            const float *e = Emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_e[kk] += val;
                sum_e_sq[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_e[kk] * sum_e[kk] - sum_e_sq[kk];
        }
        score += 0.5f * inter;
        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(sum_e); free(sum_e_sq);
    return 0;
}
```