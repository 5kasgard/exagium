```hypothesis
Family: label-fm (Factorization Machine). Learns a bias per feature value and low-rank embeddings for pairwise interactions via the FM trick: sum_{i<j} <v_i, v_j> = 0.5*( (sum v_i)^2 - sum v_i^2 ). With d=14, k=8, this adds only ~32k parameters (vs linear's ~3.5k one-hot weights) but captures all pairwise interactions efficiently. Should beat label-linear (0.8524) on tabular data with mixed categorical/ordinal features where interactions matter (e.g., education x occupation). If dev accuracy <= 0.8524, the interaction modeling is not helping.
```
```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 bytes as a categorical variable (0..255). Learns:
  w0 (scalar), w[d][256] (per-value bias), v[d][256][k] (embeddings, k=8).
Logit = w0 + sum_j w[j][x_j] + 0.5*( (sum_j v[j][x_j])^2 - sum_j v[j][x_j]^2 ).
Binary cross-entropy, Adam lr=0.01, L2=1e-4, batch=64, epochs=10.
Weights file: magic 0x464D4C31, int32 d,k; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10
K = 8  # embedding dimension


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    assert d == 14
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    idx = np.arange(n)
    for epoch in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = X[batch_idx]
            yb = y[batch_idx]
            bsz = len(yb)

            # forward
            # linear part
            lin = w0 + np.sum(w[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
            # interaction part via FM trick
            v_sum = np.sum(v[np.arange(d)[:, None], xb.T], axis=0)  # (bsz, K)
            v_sq_sum = np.sum(v[np.arange(d)[:, None], xb.T] ** 2, axis=0)  # (bsz, K)
            inter = 0.5 * np.sum(v_sum ** 2 - v_sq_sum, axis=1)  # (bsz,)
            logit = lin + inter

            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (bsz,)

            # gradients
            # w0
            gw0 = grad.mean() + L2 * w0
            # w
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], xb[:, j], grad)
            gw = gw / bsz + L2 * w
            # v
            gv = np.zeros_like(v)
            # For each sample, dlogit/dv[j][x_j] = (v_sum - v[j][x_j]) * grad
            # We can compute per feature
            for j in range(d):
                vj_x = v[j, xb[:, j]]  # (bsz, K)
                # (v_sum - vj_x) * grad[:, None]
                gv_j = (v_sum - vj_x) * grad[:, None]
                np.add.at(gv[j], xb[:, j], gv_j)
            gv = gv / bsz + L2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * gv * gv
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # write weights
    # magic, d, K
    out = struct.pack("<3i", 0x464D4C31, d, K)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, int32 K; float32 w0, w[d*256], v[d*256*K].
 * Logit = w0 + sum_j w[j][x_j] + 0.5*( (sum_j v_j)^2 - sum_j v_j^2 ). Class = logit > 0. */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], K = hdr[2];
    if (d != 14 || K != 8) { fprintf(stderr, "unexpected d/K\n"); return 3; }
    size_t expected = 12 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 12);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *v_sum = malloc(K * sizeof(float));
    float *v_sq_sum = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        for (int j = 0; j < K; j++) { v_sum[j] = 0.0f; v_sq_sum[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            logit += w[j * 256 + val];
            const float *vj = v + (j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float vv = vj[kk];
                v_sum[kk] += vv;
                v_sq_sum[kk] += vv * vv;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            logit += 0.5f * (v_sum[kk] * v_sum[kk] - v_sq_sum[kk]);
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```