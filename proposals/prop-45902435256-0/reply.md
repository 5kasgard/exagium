```hypothesis
Family: label-fm (Factorization Machine). 
Why: Adult has 14 mixed-type features with known pairwise interactions (e.g., education × occupation, marital-status × sex). 
A degree-2 FM captures all pairwise interactions via low-rank embeddings (k=8) with only ~14k parameters, 
giving richer modeling than linear at near-linear inference cost (91 dot products per item). 
Tuned on the last 20% of train (k=8, lr=0.01, 15 epochs, batch=256, L2=1e-4). 
Failure signal: dev accuracy ≤ 0.852 (label-linear) or inference > 1500 Ir/item.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights: magic 0x464D3100, d, k, C, float w0, float w[d][256], float v[d][256][k] (all little-endian float32)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 15


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
    # Convert labels to {-1, 1}
    y = np.where(y == 0, -1.0, 1.0).astype(np.float32)

    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Precompute feature indices for speed
    # For each sample, we need the 14 feature values
    # We'll process in minibatches

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsz = len(idx)

            # Forward pass
            # linear part: w0 + sum_j w[j, x_j]
            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (b,)
            # interaction part: sum_{j1<j2} <v[j1,x_j1], v[j2,x_j2]>
            # Compute sum of embeddings and sum of squares
            emb = v[np.arange(d), xb.T]  # (d, b, k)
            sum_emb = np.sum(emb, axis=0)          # (b, k)
            sum_sq_emb = np.sum(emb * emb, axis=0) # (b, k)
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)  # (b,)

            score = lin + inter                    # (b,)
            # Logistic loss gradient: dL/dscore = -y * sigmoid(-y * score)
            z = -yb * score
            sig = 1.0 / (1.0 + np.exp(-z))
            grad_score = -yb * sig                 # (b,)

            # Backward
            # w0
            g_w0 = np.mean(grad_score) + L2 * w0
            # w
            g_w = np.zeros_like(w)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, grad_score)
            g_w /= bsz
            g_w += L2 * w
            # v
            g_v = np.zeros_like(v)
            # For each feature j, gradient involves sum over other features' embeddings
            # d(inter)/d v[j, val] = sum_{l != j} v[l, x_l]
            # We can compute total_sum_emb = sum_emb (b, k)
            # For each j, other_sum = total_sum_emb - emb[j]
            for j in range(d):
                other = sum_emb - emb[j]  # (b, k)
                vals = xb[:, j]
                # g_v[j, val] += sum_b grad_score[b] * other[b]
                for k_dim in range(K):
                    np.add.at(g_v[j, :, k_dim], vals, grad_score * other[:, k_dim])
            g_v /= bsz
            g_v += L2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_hat = m_v / (1 - beta1 ** t)
            v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    # magic, d, k, C
    C = 2
    out = struct.pack("<4i", 0x464D3100, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0 or 1).
 * Weights: magic 0x464D3100, d, k, C, float w0, float w[d][256], float v[d][256][k]. */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }

    float *p = (float *)(hd + 4);
    float w0 = p[0];
    float *w = p + 1;                          // d * 256
    float *v = w + (long)d * 256;              // d * 256 * K

    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sq_emb = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear part
        float score = w0;
        for (int j = 0; j < d; j++) score += w[(long)j * 256 + xi[j]];

        // interaction part: 0.5 * ( (sum v)^2 - sum(v^2) )
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sq_emb, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *vj = v + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = vj[kk];
                sum_emb[kk] += val;
                sq_emb[kk] += val * val;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            score += 0.5f * (sum_emb[kk] * sum_emb[kk] - sq_emb[kk]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```