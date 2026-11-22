```hypothesis
Family: label-fm (Factorization Machine for tabular classification).
Why: FMs capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult where combinations of categorical features (e.g., education × occupation) strongly predict income. With 14 fields and 256 values each, a rank-4 FM adds only ~18K parameters (vs. ~100K for label-mlp16) and computes in O(d·K) per item, keeping inference fast. Fixed hyperparameters (K=4, Adam lr=3e-3, L2=1e-4) avoid tuning on the dev set.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item exceed label-linear by >20%, the interaction benefit does not justify the cost.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x464D4C31, int32 d, K, C; float32 bias, linear[d][256], emb[d][256][K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 4  # embedding rank


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    # Parameters: bias, linear[d,256], emb[d,256,K]
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (1 + K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            emb_vals = emb[np.arange(d), xb.T]  # (d, B, K)
            sum_e = np.sum(emb_vals, axis=0)    # (B, K)
            sum_e_sq = np.sum(emb_vals * emb_vals, axis=0)  # (B, K)
            inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)  # (B,)
            logits = lin_part + inter  # (B,)
            p = sigmoid(logits)

            # Gradients (binary cross-entropy)
            grad_logit = (p - yb).astype(np.float32)  # (B,)
            # bias
            g_bias = np.mean(grad_logit)
            # linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad_logit)
            g_lin /= B
            g_lin += L2 * linear
            # emb
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                e_j = emb[j, vals]  # (B, K)
                # d(inter)/d e_j = sum_e - e_j
                grad_e = (grad_logit[:, None] * (sum_e - e_j)).astype(np.float32)  # (B, K)
                np.add.at(g_emb[j], vals, grad_e)
            g_emb /= B
            g_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(np.array([bias], dtype="<f4").tobytes())
        f.write(linear.astype("<f4").tobytes())
        f.write(emb.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    if (C != 2 || K < 1 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int kk = 0; kk < K; kk++) sum_e[kk] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) sum_e[kk] += e[kk];
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float e_kk = sum_e[kk];
            inter += 0.5f * (e_kk * e_kk);
        }
        logit += inter;

        /* Two-class scores: [-logit, logit] */
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(sum_e);
    return 0;
}
```