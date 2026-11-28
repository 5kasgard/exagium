```hypothesis
Family: label-fm (factorization machine with embeddings). Adult has 14 features (6 ordinal quantized to 255 bins, 8 categorical) where pairwise interactions matter (e.g., education × occupation). A factorization machine captures these with low-rank embeddings (k=4) while keeping inference extremely cheap: O(d·k) per item. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at a fraction of the CPU cost. If dev accuracy stays ≤0.8524 or instructions/item exceed label-linear by >2×, the idea fails.
```
```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x464D3131, d, k, C, bias, linear[d][256], embed[d][256][k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
K = 4                # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30

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
    # Parameters: bias, linear[d][256], embed[d][256][K]
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed)
    v_emb = np.zeros_like(embed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = xb.shape[0]

            # Forward
            logit = bias + np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)  # (bs,)
            # Interaction term
            s_emb = np.sum(embed[np.arange(d)[:, None], xb.T], axis=0)  # (bs, K)
            sq_sum = np.sum(s_emb ** 2, axis=1)
            sum_sq = np.sum(np.sum(embed[np.arange(d)[:, None], xb.T] ** 2, axis=2), axis=0)
            inter = 0.5 * (sq_sum - sum_sq)
            logit += inter

            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (bs,)

            # Gradients
            dbias = grad.mean()
            # linear gradient
            dlinear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(dlinear[j], vals, grad)
            dlinear /= bs
            dlinear += L2 * linear

            # embedding gradient
            dembed = np.zeros_like(embed)
            for j in range(d):
                vals = xb[:, j]
                vj = embed[j, vals]  # (bs, K)
                # s_emb - vj
                diff = s_emb - vj
                # grad[:, None] * diff
                g = grad[:, None] * diff
                np.add.at(dembed[j], vals, g)
            dembed /= bs
            dembed += L2 * embed

            t += 1
            # Adam updates
            m_bias = beta1 * m_bias + (1 - beta1) * dbias
            v_bias = beta2 * v_bias + (1 - beta2) * (dbias * dbias)
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * dlinear
            v_lin = beta2 * v_lin + (1 - beta2) * (dlinear * dlinear)
            m_hat_lin = m_lin / (1 - beta1 ** t)
            v_hat_lin = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat_lin / (np.sqrt(v_hat_lin) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * dembed
            v_emb = beta2 * v_emb + (1 - beta2) * (dembed * dembed)
            m_hat_emb = m_emb / (1 - beta1 ** t)
            v_hat_emb = v_emb / (1 - beta2 ** t)
            embed -= LR * m_hat_emb / (np.sqrt(v_hat_emb) + eps)

    # Write weights
    magic = 0x464D3131
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0/1). */
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
    if (nw < 20 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + (long)d * 256 * 4 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5); // after magic,d,K,C,bias
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_emb = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        // linear part
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }
        // interaction part
        memset(s_emb, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *ej = embed + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) s_emb[f] += ej[f];
        }
        float sq_sum = 0.0f, sum_sq = 0.0f;
        for (int f = 0; f < K; f++) sq_sum += s_emb[f] * s_emb[f];
        for (int j = 0; j < d; j++) {
            const float *ej = embed + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) sum_sq += ej[f] * ej[f];
        }
        logit += 0.5f * (sq_sum - sum_sq);

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(s_emb);
    return 0;
}
```