```hypothesis
Factorization Machine (FM) with k=4 learns all pairwise feature interactions via low-rank embeddings, capturing non-linearities that label-linear misses. The fast O(d*k) inference trick (sum_v^2 - sum_norm_sq) keeps CPU cost near linear models. Expect +0.5-1.5% accuracy over label-linear at ~2x inference cost. If dev accuracy <= 0.855, the interactions aren't helping enough.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy. FM with k=4 factors per feature value, logistic loss,
Adam lr=1e-3, L2=1e-4, batch 64, epochs min(100, ceil(600000/n)). Exports: magic, d, k, C, w0, W[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # FM parameters: w0, W[d,256], V[d,256,K]
    w0 = np.float32(0.0)
    W = (rng.normal(0, 0.01, (d, 256)) * math.sqrt(2.0 / 256)).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(2.0 / 256)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx_all = np.arange(n)

    for _ in range(epochs):
        order = rng.permutation(idx_all)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = len(idx)

            # Forward
            # Linear term: w0 + sum_j W[j, xb[:,j]]
            linear = w0 + np.sum(W[np.arange(d), xb.T], axis=0)  # [B]

            # Interaction term: 0.5 * (sum_f (sum_j V[j,xb_j,f])^2 - sum_j sum_f V[j,xb_j,f]^2)
            # sum_v[f] = sum_j V[j, xb_j, f]  -> [B, K]
            sum_v = np.sum(V[np.arange(d), xb.T, :], axis=0).T   # [B, K]
            # sum_norm_sq = sum_j sum_f V[j, xb_j, f]^2 -> [B]
            sum_norm_sq = np.sum(V[np.arange(d), xb.T, :] ** 2, axis=(0, 1))  # Wait, shape
            # Fix: V[np.arange(d), xb.T, :] is [d, B, K], sum over d and K -> [B]
            sum_norm_sq = np.sum(V[np.arange(d), xb.T, :] ** 2, axis=(0, 2))

            interaction = 0.5 * (np.sum(sum_v ** 2, axis=1) - sum_norm_sq)

            logits = linear + interaction  # [B]

            # Logistic loss gradient
            # p = sigmoid(logits), grad = (p - y) / B
            max_logits = np.maximum(logits, 0)
            log_exp = np.log1p(np.exp(-np.abs(logits))) + max_logits
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb) / B  # [B]

            # Backward
            # dw0 = sum(grad)
            gw0 = float(np.sum(grad))

            # dW[j, val] = sum_{i: xb[i,j]==val} grad[i]
            gW = np.zeros_like(W)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gW[j], vals, grad)

            # dV[j, val, f] = sum_{i: xb[i,j]==val} grad[i] * (sum_v[i,f] - V[j,val,f])
            gV = np.zeros_like(V)
            # For each sample i, feature j: grad[i] * (sum_v[i] - V[j, xb[i,j], :])
            # Vectorized: for each j, scatter-add
            for j in range(d):
                vals = xb[:, j]  # [B]
                # sum_v_i = sum_v[i] for each i -> [B, K]
                # V_j_val = V[j, vals, :] -> [B, K]
                diff = sum_v - V[j, vals, :]  # [B, K]
                # grad_i * diff_i -> [B, K]
                g = (grad[:, None] * diff).astype(np.float32)  # [B, K]
                np.add.at(gV[j], vals, g)

            # Adam update
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * gw0
            v_w0 = 0.999 * v_w0 + 0.001 * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - 0.9 ** t)) / (math.sqrt(v_w0 / (1 - 0.999 ** t)) + 1e-8)

            # W
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            W -= LR * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)

            # V
            mV = 0.9 * mV + 0.1 * gV
            vV = 0.999 * vV + 0.001 * gV * gV
            V -= LR * (mV / (1 - 0.9 ** t)) / (np.sqrt(vV / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    # Magic: 0x464D3100 ("FM1\0"), d, K, C
    # Then w0 (1), W[d*256], V[d*256*K] all float32 little-endian
    magic = 0x464D3100
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr
    out += struct.pack("<f", w0)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (feature values 0-255). Output: 1 byte class (0/1).
 * Weights: magic 0x464D3100, int32 d, K, C; float32 w0, W[d][256], V[d][256][K]. */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3100) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0 || K > 16) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(wb + 16);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear term
        float linear = w0;
        for (int j = 0; j < d; j++) linear += W[(long)j * 256 + xi[j]];

        // Interaction term: 0.5 * (sum_f sum_v[f]^2 - sum_norm_sq)
        for (int f = 0; f < K; f++) sum_v[f] = 0.0f;
        float sum_norm_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = vj[f];
                sum_v[f] += v;
                sum_norm_sq += v * v;
            }
        }
        float sum_sq = 0.0f;
        for (int f = 0; f < K; f++) sum_sq += sum_v[f] * sum_v[f];
        float interaction = 0.5f * (sum_sq - sum_norm_sq);

        float logit = linear + interaction;
        float p1 = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (p1 > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```