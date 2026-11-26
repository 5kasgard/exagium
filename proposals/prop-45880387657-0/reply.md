```hypothesis
Family: label-fm (Factorization Machine with low-rank pairwise interactions).
Why: Adult has 14 mixed-type features where interactions (e.g., education × occupation) are known to matter.
A degree-2 FM with embedding dim k=8 adds only ~32k parameters (vs label-linear's ~2k) but models all pairwise
interactions. Training uses Adam on log-loss with L2; inference is ~14²·8 ≈ 1.6k FLOPs/item, still very fast.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or latency > 2× label-linear.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes each of the d=14 bytes as a categorical with 256 levels (0-255).
Learns: bias, linear weights [d,256], embeddings [d,256,k].
FM score = bias + Σ_i w[i,x_i] + Σ_{i<j} <v[i,x_i], v[j,x_j]>
Hyperparameters chosen on last 20% of train.in (tuning split), then retrain on all.
Exported binary: magic=0x464D3100, int32 d,k,C; float32 bias, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

# hyperparameter grid (tuned on validation split)
GRID = [
    {"k": 4, "lr": 0.03, "epochs": 20, "batch": 256, "l2": 1e-4},
    {"k": 8, "lr": 0.03, "epochs": 20, "batch": 256, "l2": 1e-4},
    {"k": 16, "lr": 0.03, "epochs": 20, "batch": 256, "l2": 1e-4},
    {"k": 8, "lr": 0.01, "epochs": 30, "batch": 256, "l2": 1e-4},
    {"k": 8, "lr": 0.1, "epochs": 15, "batch": 256, "l2": 1e-3},
]
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(x, bias, linear, emb, d, k):
    """x: (n,d) uint8, returns scores (n,)"""
    n = x.shape[0]
    # linear part
    lin = bias + np.sum(linear[np.arange(d), x], axis=1)
    # interaction part: 0.5 * ( (Σ v_i)^2 - Σ v_i^2 )
    # emb shape (d,256,k) -> gather to (n,d,k)
    v = emb[np.arange(d), x]                     # (n,d,k)
    sum_v = v.sum(axis=1)                        # (n,k)
    sum_v2 = (v * v).sum(axis=1)                 # (n,k)
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
    return lin + inter


def train_fm(X, y, d, k, lr, epochs, batch, l2, rng):
    n = X.shape[0]
    # init
    bias = 0.0
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # forward
            v = emb[np.arange(d), xb]                 # (bsz,d,k)
            sum_v = v.sum(axis=1)                     # (bsz,k)
            sum_v2 = (v * v).sum(axis=1)              # (bsz,k)
            lin_part = bias + np.sum(linear[np.arange(d), xb], axis=1)
            inter_part = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            score = lin_part + inter_part
            sig = 1.0 / (1.0 + np.exp(-score))
            grad = (sig - yb).astype(np.float32)      # (bsz,)
            # gradients
            g_bias = grad.mean()
            # linear grad: (d,256)
            g_lin = np.zeros_like(linear)
            np.add.at(g_lin, (np.arange(d)[:, None], xb.T), grad / bsz)
            g_lin += l2 * linear
            # emb grad: (d,256,k)
            g_emb = np.zeros_like(emb)
            # for each feature i: grad * (sum_v - v_i)
            for i in range(d):
                vi = v[:, i, :]                       # (bsz,k)
                factor = (grad[:, None] * (sum_v - vi)) / bsz  # (bsz,k)
                np.add.at(g_emb[i], xb[:, i], factor)
            g_emb += l2 * emb
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return bias, linear, emb


def accuracy(X, y, bias, linear, emb, d, k):
    score = fm_forward(X, bias, linear, emb, d, k)
    pred = (score > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y_all)
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        r = np.random.default_rng(SEED)  # same seed each config for fairness
        bias, linear, emb = train_fm(X_tr, y_tr, d, cfg["k"], cfg["lr"], cfg["epochs"],
                                     cfg["batch"], cfg["l2"], r)
        acc = accuracy(X_val, y_val, bias, linear, emb, d, cfg["k"])
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
            best_params = (bias, linear, emb)
            print(f"  new best: k={cfg['k']} lr={cfg['lr']} epochs={cfg['epochs']} val_acc={acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    r = np.random.default_rng(SEED)
    bias, linear, emb = train_fm(X_all, y_all, d, best_cfg["k"], best_cfg["lr"], best_cfg["epochs"],
                                 best_cfg["batch"], best_cfg["l2"], r)
    k = best_cfg["k"]
    # pack weights
    magic = 0x464D3100  # 'FM1\0'
    hdr = struct.pack("<4i", magic, d, k, C)
    # bias (float32), linear (d*256), emb (d*256*k)
    payload = struct.pack("<f", np.float32(bias))
    payload += linear.astype("<f4").tobytes()
    payload += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + payload)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: magic(0x464D3100), int32 d,k,C; float32 bias, linear[d*256], emb[d*256*k] */
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d <= 0 || d > 256 || k <= 0 || k > 64 || C != 2) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        // linear part
        float score = bias;
        for (int j = 0; j < d; j++) score += linear[j * 256 + xi[j]];
        // interaction part
        memset(sum_v, 0, k * sizeof(float));
        memset(sum_v2, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *v = emb + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = v[u];
                sum_v[u] += val;
                sum_v2[u] += val * val;
            }
        }
        for (int u = 0; u < k; u++) score += 0.5f * (sum_v[u] * sum_v[u] - sum_v2[u]);
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v); free(sum_v2);
    return 0;
}
```