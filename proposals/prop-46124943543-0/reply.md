```hypothesis
Family: label-fm (Factorization Machine with low-rank pairwise interactions).
Why: Adult has strong feature interactions (e.g., education×occupation, marital-status×capital-gain) that linear models miss. A factorization machine learns a scalar bias, per-value linear weights, and a K-dim embedding per feature value; pairwise interactions are computed as 0.5*(‖Σv‖²−Σ‖v‖²) in O(d·K) time. With d=14 and K=8 this adds only ~129 kB and ~100 FLOPs/item, so it should beat label-linear’s 0.8524 at similar CPU cost.
Failure: dev accuracy ≤ 0.8524 or instructions/item ≥ 2× label-linear.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31 ('FML1'), d, K, C;
float32 w0, w[d][256], v[d][256][K] (row-major)."""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (chosen on tuning split, last 20% of train.in)
K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-5, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    # split off last 20% for tuning (file order)
    n = X.shape[0]
    tune_sz = n // 5
    tr_idx = np.arange(n - tune_sz)
    tu_idx = np.arange(n - tune_sz, n)

    rng = np.random.default_rng(SEED)

    # simple grid search on tuning split
    best_acc = -1.0
    best_params = None
    for K_try in (4, 8, 16):
        for LR_try in (0.03, 0.01, 0.003):
            for L2_try in (1e-4, 1e-5):
                # initialize
                w0 = np.float32(0.0)
                w = np.zeros((d, 256), np.float32)
                v = (rng.normal(0, 0.01, (d, 256, K_try))).astype(np.float32)

                # Adam state
                m_w0 = v_w0 = np.float32(0.0)
                m_w = np.zeros_like(w); v_w = np.zeros_like(w)
                m_v = np.zeros_like(v); v_v = np.zeros_like(v)
                t = 0

                X_tr, y_tr = X[tr_idx], y[tr_idx]
                n_tr = len(X_tr)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))

                for _ in range(epochs):
                    order = rng.permutation(n_tr)
                    for s in range(0, n_tr, BATCH):
                        idx = order[s:s + BATCH]
                        xb, yb = X_tr[idx], y_tr[idx]
                        bsz = len(yb)

                        # forward
                        lin = w0 + np.sum(w[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
                        sum_v = np.zeros((bsz, K_try), np.float32)
                        sum_sq = np.zeros((bsz, K_try), np.float32)
                        for j in range(d):
                            v_j = v[j, xb[:, j]]          # (bsz, K)
                            sum_v += v_j
                            sum_sq += v_j * v_j
                        inter = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
                        logit = lin + inter

                        # gradients (binary cross-entropy)
                        p = 1.0 / (1.0 + np.exp(-logit))
                        grad = (p - yb).astype(np.float32)  # (bsz,)

                        gw0 = grad.mean()
                        gw = np.zeros_like(w)
                        gv = np.zeros_like(v)
                        for j in range(d):
                            vals = xb[:, j]
                            # linear part
                            np.add.at(gw[j], vals, grad)
                            # interaction part
                            v_j = v[j, vals]                 # (bsz, K)
                            other = sum_v - v_j              # (bsz, K)
                            np.add.at(gv[j], vals, grad[:, None] * other)

                        # L2 regularization
                        gw += L2_try * w
                        gv += L2_try * v

                        # Adam update
                        t += 1
                        beta1, beta2 = 0.9, 0.999
                        eps = 1e-8

                        m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
                        v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
                        m_w = beta1 * m_w + (1 - beta1) * gw
                        v_w = beta2 * v_w + (1 - beta2) * gw * gw
                        m_v = beta1 * m_v + (1 - beta1) * gv
                        v_v = beta2 * v_v + (1 - beta2) * gv * gv

                        m_w0_hat = m_w0 / (1 - beta1 ** t)
                        v_w0_hat = v_w0 / (1 - beta2 ** t)
                        m_w_hat = m_w / (1 - beta1 ** t)
                        v_w_hat = v_w / (1 - beta2 ** t)
                        m_v_hat = m_v / (1 - beta1 ** t)
                        v_v_hat = v_v / (1 - beta2 ** t)

                        w0 -= LR_try * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
                        w -= LR_try * m_w_hat / (np.sqrt(v_w_hat) + eps)
                        v -= LR_try * m_v_hat / (np.sqrt(v_v_hat) + eps)

                # evaluate on tuning split
                X_tu, y_tu = X[tu_idx], y[tu_idx]
                lin = w0 + np.sum(w[np.arange(d)[:, None], X_tu.T], axis=0)
                sum_v = np.zeros((len(X_tu), K_try), np.float32)
                sum_sq = np.zeros_like(sum_v)
                for j in range(d):
                    v_j = v[j, X_tu[:, j]]
                    sum_v += v_j
                    sum_sq += v_j * v_j
                inter = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
                logit = lin + inter
                pred = (logit > 0).astype(np.int64)
                acc = (pred == y_tu).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K_try, LR_try, L2_try, w0.copy(), w.copy(), v.copy())

    # retrain on full data with best hyper-parameters
    K_best, LR_best, L2_best, w0, w, v = best_params
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    m_w0 = v_w0 = np.float32(0.0)
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_v = np.zeros_like(v); v_v = np.zeros_like(v)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            lin = w0 + np.sum(w[np.arange(d)[:, None], xb.T], axis=0)
            sum_v = np.zeros((bsz, K_best), np.float32)
            sum_sq = np.zeros_like(sum_v)
            for j in range(d):
                v_j = v[j, xb[:, j]]
                sum_v += v_j
                sum_sq += v_j * v_j
            inter = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
            logit = lin + inter

            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)

            gw0 = grad.mean()
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gw[j], vals, grad)
                v_j = v[j, vals]
                other = sum_v - v_j
                np.add.at(gv[j], vals, grad[:, None] * other)

            gw += L2_best * w
            gv += L2_best * v

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * gv * gv

            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)

            w0 -= LR_best * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            w -= LR_best * m_w_hat / (np.sqrt(v_w_hat) + eps)
            v -= LR_best * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # write weights
    out = struct.pack("<4i", 0x464D4C31, d, K_best, C)
    out += np.array(w0, dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256L + 4L * d * 256L * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = w0;
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_sq[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            lin += w[(long)j * 256 + val];
            const float *v_j = v + ((long)j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float vv = v_j[kk];
                sum_v[kk] += vv;
                sum_sq[kk] += vv * vv;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sv = sum_v[kk];
            inter += 0.5f * (sv * sv - sum_sq[kk]);
        }
        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```