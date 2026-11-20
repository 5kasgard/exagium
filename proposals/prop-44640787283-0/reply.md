```hypothesis
Family: label-fm (factorization machine). 
Why: Adult has 14 features with mixed ordinal/categorical types; pairwise interactions are known to boost tabular accuracy. FM learns a linear weight plus a low-rank interaction per feature value (256 per column) with only O(d*k) extra parameters (k=8 → ~32k params). It should beat linear (0.8524) and approach MLP (0.8591) while keeping inference extremely fast (O(d*k) ≈ 112 FMA/item).
Failure signal: dev accuracy ≤ 0.8524 or inference cost > 2× label-linear.
```
```python
"""Factorization machine for tabular label tasks with byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encoding: each of the d=14 columns is a uint8 in [0,255] (255=unseen for categorical).
Model: y = bias + sum_i w_i[x_i] + sum_{f=1..k} 0.5*( (sum_i v_{i,f}[x_i])^2 - sum_i v_{i,f}[x_i]^2 )
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported weights.bin: magic=0x464D5F31, d, k, bias, w[d*256], V[d*256*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid searched on tuning split
GRID = {
    'k': [4, 8, 16],
    'lr': [0.03, 0.1, 0.3],
    'l2': [1e-4, 1e-3, 1e-2],
    'epochs': [10, 20, 30],
}
SEED = 1
BATCH = 256

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, bias, w, V, k):
    # X: (n, d) uint8
    n, d = X.shape
    lin = bias + np.sum(w[X.ravel()].reshape(n, d), axis=1)
    inter = np.zeros(n, np.float32)
    for f in range(k):
        Vf = V[:, f]  # (d*256,)
        vals = Vf[X.ravel()].reshape(n, d)
        s1 = np.sum(vals, axis=1)
        s2 = np.sum(vals * vals, axis=1)
        inter += 0.5 * (s1 * s1 - s2)
    return lin + inter

def train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, rng):
    n, d = X_tr.shape
    # init
    bias = np.float32(0.0)
    w = rng.normal(0, 0.01, d * 256).astype(np.float32)
    V = rng.normal(0, 0.01, (d * 256, k)).astype(np.float32)
    # Adam state
    params = [bias, w, V]
    shapes = [p.shape for p in params]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_val = -1
    best_params = None
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            n_b = len(idx)
            lin = bias + np.sum(w[Xb.ravel()].reshape(n_b, d), axis=1)
            inter = np.zeros(n_b, np.float32)
            for f in range(k):
                Vf = V[:, f]
                vals = Vf[Xb.ravel()].reshape(n_b, d)
                s1 = np.sum(vals, axis=1)
                s2 = np.sum(vals * vals, axis=1)
                inter += 0.5 * (s1 * s1 - s2)
            logits = lin + inter
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)  # (n_b,)
            # gradients
            g_bias = grad.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            # linear gradient
            for i in range(d):
                vals = Xb[:, i]
                np.add.at(g_w, i * 256 + vals, grad)
            g_w /= n_b
            # interaction gradient
            for f in range(k):
                Vf = V[:, f]
                for i in range(d):
                    vals = Xb[:, i]
                    v_feat = Vf[i * 256 + vals]
                    s1 = np.sum(v_feat)  # sum over batch? Wait, need per-sample sum.
                    # Actually we need per-sample gradient: for each sample, d(inter)/dV = (s1 - v_feat) * grad
                    # Let's compute per-sample:
                    # For each sample j, inter_j = 0.5*( (sum_i v_{i,f}[x_ij])^2 - sum_i v_{i,f}[x_ij]^2 )
                    # d(inter_j)/d v_{i,f}[x_ij] = (sum_i' v_{i',f}[x_i'j]) - v_{i,f}[x_ij] = s1_j - v_feat_j
                    # So we can compute per-sample s1.
                # Vectorized per-sample:
                Vmat = Vf[Xb.ravel()].reshape(n_b, d)  # (n_b, d)
                s1 = np.sum(Vmat, axis=1)  # (n_b,)
                for i in range(d):
                    v_feat = Vmat[:, i]
                    grad_V = (s1 - v_feat) * grad  # (n_b,)
                    np.add.at(g_V[:, f], i * 256 + Xb[:, i], grad_V)
            g_V /= n_b
            # L2 regularization
            g_w += l2 * w
            g_V += l2 * V
            # Adam update
            grads = [np.array([g_bias], np.float32), g_w, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # validation
        logits_val = fm_forward(X_val, bias, w, V, k)
        p_val = 1.0 / (1.0 + np.exp(-logits_val))
        acc = np.mean((p_val > 0.5) == y_val)
        if acc > best_val:
            best_val = acc
            best_params = (bias.copy(), w.copy(), V.copy())
    return best_params, best_val

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    # split: last 20% for tuning
    split = n - n // 5
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    # grid search
    best_score = -1
    best_cfg = None
    best_params = None
    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for epochs in GRID['epochs']:
                    params, score = train_fm(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, rng)
                    if score > best_score:
                        best_score = score
                        best_cfg = (k, lr, l2, epochs)
                        best_params = params
    print(f"Best cfg: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, epochs={best_cfg[3]}, val_acc={best_score:.4f}", file=sys.stderr)
    # retrain on full data with best config
    k, lr, l2, epochs = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for deterministic final training
    final_params, _ = train_fm(X, y, X_val[:1], y_val[:1], k, lr, l2, epochs, rng)  # dummy val
    bias, w, V = final_params
    # write weights.bin
    magic = 0x464D5F31  # 'FM_1' little-endian
    out = struct.pack("<3i", magic, d, k)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (uint8 feature values). Output: 1 byte class (0/1). */
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
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D5F31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    int k = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    if (d != 14 || k < 1 || k > 32) { fprintf(stderr, "bad d/k\n"); return 3; }
    size_t expected = 12 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * k * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }
    const float *p = (const float *)(wb + 12);
    float bias = p[0];
    const float *w = p + 1;
    const float *V = w + d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld len %u != %d\n", i, len, d); return 3; }
        float score = bias;
        // linear part
        for (int j = 0; j < d; j++) {
            score += w[j * 256 + xi[j]];
        }
        // interaction part
        for (int f = 0; f < k; f++) {
            float sum = 0.0f, sum_sq = 0.0f;
            const float *Vf = V + f;  // stride d*256
            for (int j = 0; j < d; j++) {
                float v = Vf[(j * 256 + xi[j]) * k];
                sum += v;
                sum_sq += v * v;
            }
            score += 0.5f * (sum * sum - sum_sq);
        }
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```