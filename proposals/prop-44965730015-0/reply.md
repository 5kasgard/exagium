```hypothesis
Family: label-fm (factorization machine with rank-4 embeddings).
Why: Adult income prediction is known to benefit from feature interactions (e.g., education × occupation). A rank-4 FM adds all pairwise interactions via low-rank embeddings, giving far more expressivity than linear with only ~18k parameters. Inference requires one pass over 14 features: accumulate linear weights and embedding vectors, then compute 0.5*(|sum|^2 - sum|vec|^2). This is ~14*4=56 FMACs + 14 loads, faster than label-mlp16 (16 hidden units) and competitive with label-linear.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ label-mlp16.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Learns per-feature-value linear weights and rank-k interaction embeddings.
Exports: magic(0x464D5442), d, k, C, W_lin[d*256], V[d*256*k], bias[1] (all float32)."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    assert C == 2, "label-fm currently supports binary classification only"

    rng = np.random.default_rng(SEED)
    # parameters: W_lin[d,256], V[d,256,K], bias[1]
    W_lin = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    bias = np.array([0.0], dtype=np.float32)

    params = [W_lin, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # forward
            linear = bias + np.sum(W_lin[np.arange(d), xb.T], axis=0)  # (B,)
            emb_sum = np.sum(V[np.arange(d), xb.T], axis=0)            # (B, K)
            square_sum = np.sum(emb_sum * emb_sum, axis=1)             # (B,)
            sum_square = np.sum(np.sum(V[np.arange(d), xb.T] ** 2, axis=2), axis=0)  # (B,)
            interaction = 0.5 * (square_sum - sum_square)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)   # (B,)

            # gradients
            g_bias = np.sum(grad_logit)
            g_W_lin = np.zeros_like(W_lin)
            g_V = np.zeros_like(V)

            # vectorized gradient accumulation
            for j in range(d):
                vals = xb[:, j]
                # linear gradient
                np.add.at(g_W_lin[j], vals, grad_logit)
                # interaction gradient: grad_logit * (emb_sum - V_j)
                v_j = V[j, vals]                 # (B, K)
                diff = emb_sum - v_j             # (B, K)
                g_V_j = grad_logit[:, None] * diff
                np.add.at(g_V[j], vals, g_V_j)

            # L2 regularization
            g_W_lin += L2 * W_lin
            g_V += L2 * V

            grads = [g_W_lin, g_V, np.array([g_bias], dtype=np.float32)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x464D5442  # "BTMF"
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Binary classification with rank-K FM. Output: one byte per record (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D5442) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 * ((size_t)d * 256 + (size_t)d * 256 * K + 1);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *V = W_lin + (size_t)d * 256;
    const float bias = V[(size_t)d * 256 * K];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias;
        memset(emb_sum, 0, K * sizeof(float));
        float sum_square = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += W_lin[(size_t)j * 256 + v];
            const float *v_ptr = V + ((size_t)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = v_ptr[kk];
                emb_sum[kk] += val;
                sum_square += val * val;
            }
        }
        float square_sum = 0.0f;
        for (int kk = 0; kk < K; kk++) square_sum += emb_sum[kk] * emb_sum[kk];
        float interaction = 0.5f * (square_sum - sum_square);
        float logit = linear + interaction;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(emb_sum);
    return 0;
}
```