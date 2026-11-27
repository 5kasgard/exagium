```hypothesis
Factorization Machine (FM) with field-structured embeddings: each of the 14 columns gets a 256-value embedding table (K=8). The model scores bias + linear terms + pairwise dot products between column embeddings. Adult has strong pairwise interactions (education×occupation, marital×relationship) that linear models miss; FM captures them with O(d·K) params and O(d·K) inference. Target: dev accuracy > 0.8524 (label-linear). If dev ≤ 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only. Field-aware FM: 14 fields × 256 values × K=8 embeddings + linear terms.
Adam lr=0.01, L2=1e-4, batch=256, epochs=12. Binary logistic loss. Exports: magic, d, K, C, bias, linear[d*256], embed[d*256*K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 12
MAGIC = 0x464D2031  # "FM 1"

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
    assert C == 2, "FM binary only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # params: bias (1), linear (d*256), embed (d*256*K)
    bias = np.float32(0.0)
    linear = np.zeros(d * 256, np.float32)
    embed = (rng.normal(0, 0.01, (d * 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: compute logits and gradients
            # logit = bias + sum_j linear[j, x_j] + 0.5 * (||sum_j v_j||^2 - sum_j ||v_j||^2)
            # For binary: target y in {0,1}, loss = log(1+exp(-(2y-1)*logit))
            # grad_logit = -(2y-1) * sigmoid(-(2y-1)*logit) = sigmoid(logit) - y

            # gather embeddings for batch: [bsz, d, K]
            # flat indices: xb[:,j] + j*256
            flat_idx = xb + (np.arange(d, dtype=np.int64) * 256)
            emb_batch = embed[flat_idx]  # [bsz, d, K]
            lin_batch = linear[flat_idx]  # [bsz, d]

            sum_emb = emb_batch.sum(axis=1)  # [bsz, K]
            sum_emb_sq = (emb_batch * emb_batch).sum(axis=1)  # [bsz, K]
            inter = 0.5 * ((sum_emb * sum_emb).sum(axis=1) - sum_emb_sq.sum(axis=1))  # [bsz]
            lin_sum = lin_batch.sum(axis=1)  # [bsz]
            logits = bias + lin_sum + inter  # [bsz]

            probs = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (probs - yb).astype(np.float32)  # [bsz]

            # gradients
            g_bias = grad_logit.mean()
            # linear grad: for each j, grad_logit[:,None] added to linear[j, x_j]
            g_lin = np.zeros_like(linear)
            np.add.at(g_lin, flat_idx.ravel(), np.repeat(grad_logit, d) / bsz)

            # embedding grad: for each sample, each field j:
            # d(inter)/d v_j = (sum_emb - v_j) * grad_logit
            # sum_emb is [bsz, K], v_j is [bsz, K]
            diff = (sum_emb[:, None, :] - emb_batch) * grad_logit[:, None, None]  # [bsz, d, K]
            g_emb = np.zeros_like(embed)
            np.add.at(g_emb, flat_idx.ravel(), diff.reshape(-1, K) / bsz)

            # L2 regularization on linear and embed (not bias)
            g_lin += L2 * linear
            g_emb += L2 * embed

            t += 1
            # Adam update
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (math.sqrt(v_bias / (1 - beta2 ** t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # pack weights
    out = struct.pack("<4if", MAGIC, d, K, C, bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class (0 or 1).
 * Weights: magic, d, K, C, bias, linear[d*256], embed[d*256*K] (all float32 little-endian). */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D2031) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    int K = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    int C = wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24;
    if (C != 2 || d != 14 || K <= 0 || K > 64) { fprintf(stderr, "bad header\n"); return 3; }

    const float *p = (const float *)(wb + 16);
    float bias = *p++;
    long lin_sz = (long)d * 256;
    long emb_sz = (long)d * 256 * K;
    if (nw != 16 + 4L * (1 + lin_sz + emb_sz)) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = p; p += lin_sz;
    const float *embed = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(K, sizeof(float));
    float *sum_emb_sq = calloc(K, sizeof(float));
    float *vj = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = 0.0f;
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_emb_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            int idx = xi[j] + j * 256;
            lin_sum += linear[idx];
            const float *e = embed + (long)idx * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = sum_emb[kk];
            inter += s * s - sum_emb_sq[kk];
        }
        inter *= 0.5f;

        float logit = bias + lin_sum + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```