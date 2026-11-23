```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult where combinations (e.g., education × occupation) are predictive. The model adds only O(d·k) parameters (d=14, k=8) and computes interactions in O(d·k) time, keeping inference fast. If dev accuracy does not exceed label-linear (0.8524) or inference cost is not lower than label-linear (719 Ir/item), the approach fails.
```

```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Input: 14 bytes per record (raw feature indices). Output: binary class (0/1).
Model: bias + linear weights per feature value + k-dim embeddings per feature value.
Score = bias + sum_j w[j][x_j] + 0.5 * ( (sum_j E[j][x_j])^2 - sum_j E[j][x_j]^2 ).
Deterministic Adam, seed=1, single thread."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D3130  # 'FM10'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias (1), linear (d,256), embed (d,256,K)
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (256 * K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            emb_sum = np.sum(embed[np.arange(d), xb.T], axis=0)           # (B, K)
            emb_sq_sum = np.sum(embed[np.arange(d), xb.T] ** 2, axis=0)   # (B, K)
            inter = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1)       # (B,)
            logits = lin_part + inter                                     # (B,)

            # Logistic loss gradient w.r.t logits
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)  # (B,)

            # Gradients
            g_bias = np.mean(grad)
            # Linear gradients: for each feature j, value v, sum grad over batch where xb[:,j]==v
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g_lin[j, v] = np.mean(grad[mask])
            # Embedding gradients
            g_emb = np.zeros_like(embed)
            # d(inter)/dE[j,v] = grad * (emb_sum - E[j,v]) for each sample
            # We accumulate over batch
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        # (emb_sum[mask] - embed[j,v]) shape (M, K)
                        diff = emb_sum[mask] - embed[j, v]
                        g_emb[j, v] = np.mean(grad[mask, None] * diff, axis=0)

            # Add L2 regularization (only on linear and embed, not bias)
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # embed
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            embed -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    # Header: magic, d, K, 2 (classes)
    # Then: bias (float32), linear (d*256 float32), embed (d*256*K float32)
    out = struct.pack("<4i", MAGIC, d, K, 2)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output record: 1 byte (class 0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0 || K > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int u = 0; u < K; u++) { emb_sum[u] = 0.0f; emb_sq_sum[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(long)j * 256 + v];
            const float *e = embed + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                emb_sum[u] += val;
                emb_sq_sum[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += emb_sum[u] * emb_sum[u] - emb_sq_sum[u];
        inter *= 0.5f;
        float score = lin + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```