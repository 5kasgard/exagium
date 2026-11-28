```hypothesis
Factorization Machine (degree-2) with latent dimension k=8 captures pairwise feature interactions
(e.g., education×occupation, marital-status×sex) known to be predictive on Adult, while inference
cost stays near-linear: 14 non-zero features → O(k·14) ops. Beats label-linear (0.8524) if
interactions matter; fails if signal is purely additive or if k=8 overfits the 26k train rows.
```
```python
"""Factorization Machine (degree-2) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only.
Weights format (little-endian):
  uint32 magic=0x464D3231 ("FM21"), d, k, C
  float32 bias
  float32 w[d][256]          # linear weight per (column, value)
  float32 V[d][256][k]       # latent vector per (column, value)
Inference: sum w_j[x_j] + 0.5*sum_f((sum_j V_j[x_j][f])^2 - sum_j V_j[x_j][f]^2) -> argmax over C=2.
"""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 600000, 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Build per-column encoding: for each column j, map value 0..255 -> feature index
    # We'll use a flat feature space: each (j, v) is a distinct feature.
    # Linear weights w[j, v], latent vectors V[j, v, f]
    w = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    n = X.shape[0]
    # Precompute for each sample the list of (j, v) pairs (always d pairs, one per column)
    # For ordinal columns, the paper uses quantile bins (already 0..255), treat as categorical with 256 values.
    # For categorical, 255 = unseen. We'll keep all 256 values for simplicity.
    # X is already uint8 0..255.
    cols = np.arange(d, dtype=np.int32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Linear term: sum_j w[j, xb[:, j]]
            # V term: for each f, sum_j V[j, xb[:, j], f]
            # We'll vectorize over batch and features.
            # Gather weights for this batch: w_batch[b, j] = w[j, xb[b, j]]
            w_batch = w[cols, xb.T].T                     # (B, d)
            V_batch = V[cols, xb.T].transpose(1, 0, 2)    # (B, d, K)

            linear = w_batch.sum(axis=1) + bias           # (B,)
            # FM interaction: 0.5 * sum_f ( (sum_j V_{j,f})^2 - sum_j V_{j,f}^2 )
            sum_V = V_batch.sum(axis=1)                   # (B, K)
            sum_V2 = (V_batch * V_batch).sum(axis=1)      # (B, K)
            inter = 0.5 * ((sum_V * sum_V - sum_V2).sum(axis=1))  # (B,)

            logits = linear + inter                       # (B,)
            # Binary logistic loss gradient w.r.t logits: sigmoid(logits) - y
            # For C=2, we model P(y=1) = sigmoid(logits), P(y=0) = 1-sigmoid
            # But the task expects class 0 or 1. We'll train binary.
            # Sigmoid
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)            # (B,)

            # Gradients
            # dL/dw[j,v] = sum_{b: xb[b,j]=v} grad[b] + L2 * w[j,v]
            # dL/dV[j,v,f] = sum_{b: xb[b,j]=v} grad[b] * (sum_V[b,f] - V[j,v,f]) + L2 * V[j,v,f]
            # dbias = sum grad[b]

            # Accumulate per (j,v)
            # We'll use a scatter-add via looping over j (d=14 small)
            dw = np.zeros_like(w)
            dV = np.zeros_like(V)
            db = grad.sum()

            for j in range(d):
                vals = xb[:, j]                           # (B,)
                # For each unique value in this batch column
                uniq, inv = np.unique(vals, return_inverse=True)
                # grad per sample
                g = grad                                   # (B,)
                # dw[j, v] = sum_{b: vals[b]=v} g[b] + L2*w
                # Use bincount
                dw_j = np.bincount(vals, weights=g, minlength=256)
                dw[j] = dw_j + L2 * w[j]

                # dV: for each f, sum_b g[b] * (sum_V[b,f] - V[j, vals[b], f])
                # sum_V is (B, K), V[j, vals] is (B, K)
                diff = sum_V - V[j, vals]                 # (B, K)
                # g[:, None] * diff -> (B, K), sum over b per value
                for f in range(K):
                    dV[j, :, f] = np.bincount(vals, weights=g * diff[:, f], minlength=256) + L2 * V[j, :, f]

            # Adam update
            t += 1
            # bias
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * (db * db)
            bias -= LR * (m_b / (1 - beta1 ** t)) / (math.sqrt(v_b / (1 - beta2 ** t)) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw * dw)
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV * dV)
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Save weights
    # magic, d, K, C
    out = struct.pack("<4I", 0x464D3231, d, K, C)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   uint32 magic=0x464D3231, d, k, C
 *   float32 bias
 *   float32 w[d][256]
 *   float32 V[d][256][k]
 * Input record: d bytes (one per column). Output: 1 byte class (0 or 1).
 */
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
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[8], sum_V2[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = *bias;
        for (int f = 0; f < K; f++) { sum_V[f] = 0.0f; sum_V2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            const float *w_jv = w + ((long)j * 256 + v);
            linear += *w_jv;
            const float *V_jv = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = V_jv[f];
                sum_V[f] += val;
                sum_V2[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += sum_V[f] * sum_V[f] - sum_V2[f];
        }
        inter *= 0.5f;

        float logit = linear + inter;
        float p1 = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (p1 > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```