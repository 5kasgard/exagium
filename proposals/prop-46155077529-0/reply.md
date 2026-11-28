```hypothesis
Family: label-fm (Factorization Machine with binary features).
Adult has many categorical features where pairwise interactions (e.g., education × occupation) are predictive. A rank-8 FM adds only ~29k parameters (vs ~3.6k for linear) and computes interactions in O(d·k) at inference. It should beat label-linear (0.8524) while staying faster than label-mlp16. If dev accuracy ≤ 0.8524 or instructions/item exceed label-mlp16, the idea fails.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            linear = w0 + np.sum(w[np.arange(d), Xb.T], axis=0)   # (B,)
            V_sel = V[np.arange(d), Xb.T]                         # (d, B, K)
            sum_v = np.sum(V_sel, axis=0)                         # (B, K)
            sum_sq = np.sum(V_sel * V_sel, axis=0)                # (B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)  # (B,)
            logit = linear + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)              # (B,)

            # Gradients
            gw0 = grad_logit.mean()
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(gw[j], vals, grad_logit)
                diff = grad_logit[:, None] * (sum_v - V_sel[j])   # (B, K)
                for u in range(K):
                    np.add.at(gV[j, :, u], vals, diff[:, u])
            gw /= B
            gV /= B

            # L2
            gw += L2 * w
            gV += L2 * V

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m0_hat = m_w0 / (1 - beta1 ** t)
            v0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m0_hat / (np.sqrt(v0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            mV_hat = m_V / (1 - beta1 ** t)
            vV_hat = v_V / (1 - beta2 ** t)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + eps)

    # Pack weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class (0/1). */
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
    if (C != 2 || d < 1 || K < 1 || K > 64) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(K * sizeof(float));
    float *sum_sq = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[j * 256 + val];
            const float *v = V + (j * 256 + val) * K;
            for (int u = 0; u < K; u++) {
                float vv = v[u];
                sum_v[u] += vv;
                sum_sq[u] += vv * vv;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            inter += sum_v[u] * sum_v[u] - sum_sq[u];
        }
        inter *= 0.5f;

        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_v); free(sum_sq);
    return 0;
}
```