```hypothesis
The family "label-fm" implements a factorization machine with per-class embeddings (K=4) on the raw 256-value bytes.
Pairwise interactions are crucial for tabular data like Adult; the FM captures them in O(d·C·K) time, adding only ~100 FLOPs/item over linear.
If dev accuracy does not exceed label-linear (0.8524), it indicates the pre-binned representation already linearizes the signal or that K=4 is too small.
```
```python
"""label-fm: Factorization Machine with per-class embeddings for tabular byte features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Chooses embedding dim K on tuning split (last 20%), then trains on all data.
Weights format: magic=0x464D4C31, d, K, C; float32 linear[d][256][C], fm[d][256][K*C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [0, 2, 4, 8]  # K=0 is pure linear

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, K, rng, n_epochs):
    n, d = X.shape
    # Parameters: linear[d,256,C], fm[d,256,K,C], bias[C]
    linear = np.zeros((d, 256, C), np.float32)
    fm = np.zeros((d, 256, K, C), np.float32) if K else None
    bias = np.zeros(C, np.float32)
    params = [linear, fm, bias] if K else [linear, bias]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params if p is not None]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # Forward
            lin_score = bias.copy()
            fm_sum = np.zeros((bsz, K, C), np.float32) if K else None
            fm_sq = np.zeros((bsz, K, C), np.float32) if K else None
            for j in range(d):
                v = xb[:, j]
                lin_score += linear[j, v]
                if K:
                    fm_j = fm[j, v]  # (bsz, K, C)
                    fm_sum += fm_j
                    fm_sq += fm_j * fm_j
            if K:
                inter = 0.5 * ((fm_sum * fm_sum).sum(axis=1) - fm_sq.sum(axis=1))  # (bsz, C)
                score = lin_score + inter
            else:
                score = lin_score
            # Softmax cross-entropy gradient
            score -= score.max(axis=1, keepdims=True)
            exp_s = np.exp(score)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad = probs / bsz  # (bsz, C)
            # Backward
            grad_linear = np.zeros_like(linear)
            grad_fm = np.zeros_like(fm) if K else None
            grad_bias = grad.sum(axis=0)
            for j in range(d):
                v = xb[:, j]
                np.add.at(grad_linear[j], v, grad)
                if K:
                    fm_j = fm[j, v]  # (bsz, K, C)
                    # dL/d fm_j = grad[:,None,:] * (fm_sum - fm_j)
                    # fm_sum is (bsz,K,C), fm_j is (bsz,K,C)
                    delta = grad[:, None, :] * (fm_sum - fm_j)
                    np.add.at(grad_fm[j], v, delta)
            # L2 regularization
            grad_linear += L2 * linear
            grad_bias += L2 * bias
            if K:
                grad_fm += L2 * fm
            # Adam update
            t += 1
            grads = [grad_linear, grad_fm, grad_bias] if K else [grad_linear, grad_bias]
            for i, (p, gr) in enumerate(zip(params, grads)):
                if p is None: continue
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return linear, fm, bias

def evaluate(X, y, C, linear, fm, bias, K):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        lin_score = bias.copy()
        fm_sum = np.zeros((K, C), np.float32) if K else None
        fm_sq = np.zeros((K, C), np.float32) if K else None
        for j in range(d):
            v = X[i, j]
            lin_score += linear[j, v]
            if K:
                fm_j = fm[j, v]
                fm_sum += fm_j
                fm_sq += fm_j * fm_j
        if K:
            inter = 0.5 * ((fm_sum * fm_sum).sum(axis=0) - fm_sq.sum(axis=0))
            score = lin_score + inter
        else:
            score = lin_score
        pred = np.argmax(score)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]
    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    best_K, best_acc = 0, -1.0
    for K in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + K)
        linear, fm, bias = train_model(X_train, y_train, C, K, rng_k, n_epochs)
        acc = evaluate(X_tune, y_tune, C, linear, fm, bias, K)
        if acc > best_acc:
            best_acc, best_K = acc, K
    # Retrain on full data with best_K
    rng_full = np.random.default_rng(SEED + 100 + best_K)
    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    linear, fm, bias = train_model(X_all, y_all, C, best_K, rng_full, n_epochs_full)
    # Write weights.bin
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, best_K, C)
    out += linear.astype("<f4").tobytes()
    if best_K:
        out += fm.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights: magic 0x464D4C31, int32 d, K, C; float32 linear[d][256][C], fm[d][256][K*C] (if K>0), bias[C]. */
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
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    size_t linear_sz = (size_t)d * 256 * C;
    size_t fm_sz = K ? (size_t)d * 256 * K * C : 0;
    size_t bias_sz = C;
    size_t total_floats = linear_sz + fm_sz + bias_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *fm = linear + linear_sz;
    const float *bias = fm + fm_sz;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    float *fm_sum = K ? malloc(sizeof(float) * K * C) : NULL;
    float *fm_sq  = K ? malloc(sizeof(float) * K * C) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(score, bias, sizeof(float) * C);
        if (K) memset(fm_sum, 0, sizeof(float) * K * C), memset(fm_sq, 0, sizeof(float) * K * C);
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin = linear + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) score[c] += lin[c];
            if (K) {
                const float *emb = fm + ((size_t)j * 256 + v) * K * C;
                for (int c = 0; c < C; c++) {
                    for (int kk = 0; kk < K; kk++) {
                        float e = emb[kk * C + c];
                        fm_sum[kk * C + c] += e;
                        fm_sq[kk * C + c] += e * e;
                    }
                }
            }
        }
        if (K) {
            for (int c = 0; c < C; c++) {
                float inter = 0.0f;
                for (int kk = 0; kk < K; kk++) {
                    float s = fm_sum[kk * C + c];
                    inter += 0.5f * (s * s - fm_sq[kk * C + c]);
                }
                score[c] += inter;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```