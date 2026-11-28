```hypothesis
label-fm: Factorization Machine with 8-dim embeddings for all 14 features (treated as 256-value categorical).
Captures pairwise interactions (e.g., education×occupation) that linear models miss.
Adam lr=0.01, L2=1e-4, 10 epochs, batch 256, seed=1.
If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000, the model fails its goal.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only.
Weights format: magic 0x464D3130, d, k, 0; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 10
K = 8          # embedding dimension

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
    C = card["classes"]
    assert C == 2, "FM only for binary"
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters: w0 (scalar), w[d*256], v[d*256*K]
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    v = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0 = v_w0 = np.float32(0.0)
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_v = np.zeros_like(v); v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward_batch(xb):
        """xb: (B, d) uint8 -> logits (B,)"""
        B = xb.shape[0]
        logits = np.full(B, w0, dtype=np.float32)
        # linear part
        for j in range(d):
            logits += w[j, xb[:, j]]
        # FM pairwise part: 0.5 * ( (sum v)^2 - sum v^2 ) summed over K
        sum_v = np.zeros((B, K), dtype=np.float32)
        sum_v2 = np.zeros((B, K), dtype=np.float32)
        for j in range(d):
            vj = v[j, xb[:, j]]          # (B, K)
            sum_v += vj
            sum_v2 += vj * vj
        inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
        logits += inter
        return logits

    def loss_grad(logits, yb):
        """logits (B,), yb (B,) in {0,1} -> grad wrt logits (B,)"""
        # BCE with logits: max(logits,0) - logits*yb + log(1+exp(-|logits|))
        # derivative: sigmoid(logits) - yb
        p = 1.0 / (1.0 + np.exp(-logits))
        return (p - yb).astype(np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            logits = forward_batch(xb)
            g = loss_grad(logits, yb)          # (B,)

            # gradients
            gw0 = np.mean(g)
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)

            # linear gradients
            for j in range(d):
                vals = xb[:, j]
                for val in range(256):
                    mask = (vals == val)
                    if np.any(mask):
                        gw[j, val] = np.mean(g[mask])

            # FM gradients
            # For each example, d logits wrt v_j = g_i * (sum_v - v_j)  (element-wise over K)
            # We accumulate per (j, val)
            sum_v_batch = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                sum_v_batch += v[j, xb[:, j]]
            for j in range(d):
                vj = v[j, xb[:, j]]          # (B, K)
                # g_i[:,None] * (sum_v_batch - vj)
                grad_vj = g[:, None] * (sum_v_batch - vj)   # (B, K)
                vals = xb[:, j]
                for val in range(256):
                    mask = (vals == val)
                    if np.any(mask):
                        gv[j, val] = np.mean(grad_vj[mask], axis=0)

            # L2 regularization
            gw += L2 * w
            gv += L2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * (gv * gv)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # write weights
    magic = 0x464D3130  # 'FM10'
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<4i", magic, d, K, 0))
        f.write(struct.pack("<f", w0))
        f.write(w.astype("<f4").tobytes())
        f.write(v.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class (0 or 1). */
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
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2];
    if (d <= 0 || K <= 0 || d > 256 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;                     // scalar bias
    const float *w = p; p += d * 256;    // linear weights
    const float *v = p;                  // embeddings [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    float *sum_v2 = calloc(K, sizeof(float));
    if (!sum_v || !sum_v2) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            int val = xi[j];
            int idx = j * 256 + val;
            logit += w[idx];
            const float *vj = v + (long)idx * K;
            for (int f = 0; f < K; f++) {
                float e = vj[f];
                sum_v[f] += e;
                sum_v2[f] += e * e;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }
        unsigned char out = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out, 1);
    }
    free(sum_v); free(sum_v2);
    return 0;
}
```