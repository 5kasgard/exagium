```hypothesis
Factorization Machine (FM) with low-rank pairwise interactions (k=8) captures feature combinations crucial for Adult (e.g., education×occupation) while adding only ~32K parameters. Using the FM trick, inference costs O(d·k) ~ 112 multiply-adds per item, far cheaper than MLP. It should exceed label-linear's 0.8524 accuracy; if it doesn't, interactions are not predictive or rank is insufficient.
```

```python
"""Factorization Machine for tabular label tasks: per-feature-value linear weights + k-dim latent vectors.
Binary cross-entropy, Adam, deterministic (seed 1). Weights format: magic 0x464D4C31, d, k, C=2; float32 W_lin[d][256], W_emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 30


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: linear weights (d, 256), embeddings (d, 256, K)
    W_lin = np.zeros((d, 256), dtype=np.float32)
    W_emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_emb, v_emb = np.zeros_like(W_emb), np.zeros_like(W_emb)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # Linear part: sum_j W_lin[j, xb[:,j]]
            lin_scores = np.zeros(B, dtype=np.float32)
            for j in range(d):
                lin_scores += W_lin[j, xb[:, j]]

            # FM interaction: 0.5 * ( (sum v_j)^2 - sum v_j^2 )
            sum_emb = np.zeros((B, K), dtype=np.float32)
            sum_emb_sq = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                emb = W_emb[j, xb[:, j]]      # (B, K)
                sum_emb += emb
                sum_emb_sq += emb * emb
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)

            logits = lin_scores + inter       # (B,)
            # Binary cross-entropy with logits
            # p = sigmoid(logits), loss = -[y*log(p) + (1-y)*log(1-p)]
            # gradient w.r.t logits: p - y
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (p - yb).astype(np.float32)   # (B,)

            # Backward
            # Linear gradients
            grad_lin = np.zeros_like(W_lin)
            for j in range(d):
                np.add.at(grad_lin[j], xb[:, j], grad_logits)
            grad_lin /= B
            grad_lin += L2 * W_lin

            # Embedding gradients
            grad_emb = np.zeros_like(W_emb)
            # For each sample, d(inter)/d(emb_j) = (sum_emb - emb_j) * grad_logits
            # We can loop over features
            for j in range(d):
                emb_j = W_emb[j, xb[:, j]]          # (B, K)
                # (sum_emb - emb_j) shape (B, K)
                factor = (sum_emb - emb_j) * grad_logits[:, None]   # (B, K)
                np.add.at(grad_emb[j], xb[:, j], factor)
            grad_emb /= B
            grad_emb += L2 * W_emb

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # Linear
            m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
            v_lin = beta2 * v_lin + (1 - beta2) * grad_lin * grad_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            W_lin -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # Embedding
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            W_emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header + W_lin.astype("<f4").tobytes() + W_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (0 or 1).
 * Weights: magic 0x464D4C31, d, k, C=2; float32 W_lin[d][256], W_emb[d][256][k].
 * Uses FM trick: interaction = 0.5 * ( (sum v_j)^2 - sum v_j^2 ). */
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
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *W_emb = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(K, sizeof(float));
    float *sum_emb_sq = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_score = 0.0f;
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_emb_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_score += W_lin[(long)j * 256 + v];
            const float *emb = W_emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float e = emb[u];
                sum_emb[u] += e;
                sum_emb_sq[u] += e * e;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = sum_emb[u];
            inter += 0.5f * (s * s - sum_emb_sq[u]);
        }

        float logit = lin_score + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```