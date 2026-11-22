```hypothesis
Factorization Machine (FM) with 8-dim embeddings for pairwise feature interactions. Each of the 14 columns has 256 possible values (including unseen=255); FM learns a linear weight and k-dim vector per (column,value). Captures feature crosses critical for tabular data (e.g., education×occupation) with O(d²k) inference (d=14, k=8 → 728 mult-adds). Beats linear by modeling interactions; fails if interactions are noise or overfit dominates.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. FM: w0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j. k=8, Adam lr=1e-2, L2=1e-5, batch=256, epochs=min(50, ceil(500000/n))."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 500000, 50


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Feature indexing: each column j has 256 values -> feat_id = j*256 + val
    n_feat = d * 256

    # Parameters
    w0 = np.float32(0.0)
    w = np.zeros(n_feat, dtype=np.float32)
    v = (rng.normal(0, 0.01, (n_feat, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    # Precompute feature ids for all samples
    feat_ids = np.empty((n, d), dtype=np.int32)
    for j in range(d):
        feat_ids[:, j] = j * 256 + X[:, j]

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            ids = feat_ids[idx]           # (B, d)
            yy = y[idx]                   # (B,)

            # Linear part
            lin = w0 + np.sum(w[ids], axis=1)  # (B,)

            # FM interaction part: Σ_{i<j} <v_i, v_j>
            # Compute sum_v = Σ v_i, sum_v2 = Σ v_i^2, then 0.5*(sum_v^2 - sum_v2)
            v_batch = v[ids]              # (B, d, K)
            sum_v = np.sum(v_batch, axis=1)          # (B, K)
            sum_v2 = np.sum(v_batch * v_batch, axis=1)  # (B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (B,)

            logits = lin + inter
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yy).astype(np.float32)  # (B,)

            # Gradients
            gw0 = grad.mean()
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)

            # Linear gradient
            np.add.at(gw, ids.ravel(), grad.repeat(d) / BATCH)
            gw += L2 * w

            # Interaction gradient: for each sample, for each feature i, grad_i = grad * (sum_v - v_i)
            # Vectorized: for each feature position j in 0..d-1
            for j in range(d):
                vj = v_batch[:, j, :]           # (B, K)
                # sum_v - vj
                diff = sum_v - vj               # (B, K)
                # grad[:, None] * diff  -> (B, K)
                gvj = (grad[:, None] * diff) / BATCH
                feat_j = ids[:, j]
                np.add.at(gv, feat_j, gvj)
            gv += L2 * v

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

            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # Export weights: magic, d, K, w0, w[n_feat], v[n_feat*K]
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4i f", magic, d, K, n_feat, w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0/1). FM: w0 + Σ w_i + Σ_{i<j} <v_i,v_j>. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], n_feat = hd[3];
    if (d != 14 || n_feat != d * 256) { fprintf(stderr, "bad dims\n"); return 3; }
    float *ptr = (float *)(hd + 4);
    float w0 = *ptr++;
    float *w = ptr; ptr += n_feat;
    float *v = ptr;
    long expected = 20 + (1 + n_feat + (long)n_feat * K) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        // Linear part
        for (int j = 0; j < d; j++) {
            int fid = j * 256 + xi[j];
            score += w[fid];
        }
        // Interaction part: 0.5 * ( (Σ v_i)^2 - Σ v_i^2 )
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            int fid = j * 256 + xi[j];
            const float *vj = v + (long)fid * K;
            for (int kk = 0; kk < K; kk++) {
                float vjk = vj[kk];
                sum_v[kk] += vjk;
                sum_v2[kk] += vjk * vjk;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            score += 0.5f * (sum_v[kk] * sum_v[kk] - sum_v2[kk]);
        }
        unsigned char pred = score > 0.0f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```