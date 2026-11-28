```hypothesis
We propose a Factorization Machine family "label-fm". FM models pairwise feature interactions via low-rank embeddings, which are known to boost accuracy on tabular data like Adult (e.g., education × occupation). With latent dimension k=8, inference costs O(d·k) ≈ 112 FLOPs/item, comparable to label-linear. If dev accuracy does not exceed label-linear's 0.8524 or instructions/item exceed label-linear by >20%, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights: int32 magic 0x464D0001, d, k, C; float32 bias; per-column 256*(1+k) floats (linear + latent)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D0001

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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias (1), linear[d,256], latent[d,256,k]
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    latent = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_lat, v_lat = np.zeros_like(latent), np.zeros_like(latent)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            lin_sum = bias + np.sum(linear[np.arange(d)[:,None], xb.T], axis=0)  # (bsz,)
            sum_v = np.zeros((bsz, K), dtype=np.float32)
            sum_v_sq = np.zeros((bsz, K), dtype=np.float32)
            for j in range(d):
                v = latent[j, xb[:, j]]  # (bsz, K)
                sum_v += v
                sum_v_sq += v * v
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
            logit = lin_sum + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            g = (p - yb).astype(np.float32)  # (bsz,)

            # Gradients
            db = g.mean()
            dlin = np.zeros_like(linear)
            dlat = np.zeros_like(latent)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(dlin[j], vals, g)
                gv = g[:, None] * (sum_v - latent[j, vals])
                np.add.at(dlat[j], vals, gv)

            # L2 regularization on latent only (linear not regularized in standard FM)
            dlat += L2 * latent

            t += 1
            # Adam update
            def adam_step(param, grad, m, v):
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
                return m, v

            m_bias, v_bias = adam_step(bias, db, m_bias, v_bias)
            m_lin, v_lin = adam_step(linear, dlin, m_lin, v_lin)
            m_lat, v_lat = adam_step(latent, dlat, m_lat, v_lat)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += struct.pack("<f", bias)
    for j in range(d):
        for v in range(256):
            out += struct.pack("<f", linear[j, v])
            out += latent[j, v].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label-fm.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1). */
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
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * (1 + K) * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 16);
    float bias = *p++;
    // linear[d][256], latent[d][256][K]
    const float *linear = p; p += d * 256;
    const float *latent = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(K * sizeof(float));
    float *sum_v_sq = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = bias;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            size_t idx = j * 256 + v;
            lin_sum += linear[idx];
            const float *lv = latent + (size_t)idx * K;
            for (int f = 0; f < K; f++) {
                float val = lv[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sum_v[f] * sum_v[f] - sum_v_sq[f];
        inter *= 0.5f;
        float logit = lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```