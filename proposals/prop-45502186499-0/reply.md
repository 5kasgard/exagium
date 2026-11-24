```hypothesis
Factorization Machines (FM) capture pairwise feature interactions efficiently via low-rank embeddings, which suits tabular data like Adult where interactions (e.g., education × occupation) matter. A binary FM with k=8 adds only ~32K parameters vs linear's ~3.5K, yet models all pairwise interactions. It should beat label-linear's 0.8524 accuracy while keeping inference fast (O(d·k) per item). If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear by >2×, the hypothesis fails.
```

```python
"""Factorization Machine family for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: each of the d bytes is a categorical feature with 256 values (0-255).
Model: bias + linear weights per feature-value + pairwise interactions via k-dim embeddings.
Binary classification (sigmoid), trained with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exported weights: magic 0x464D31, d, k, C; float32 bias[1], linear[d*256], embed[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.array([0.0], dtype=np.float32)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            lin_sum = np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            # Embeddings for active values: (B, d, K)
            emb_vals = embed[np.arange(d), xb.T]                   # (d, B, K) -> transpose to (B, d, K)
            emb_vals = emb_vals.transpose(1, 0, 2)
            sum_emb = np.sum(emb_vals, axis=1)                     # (B, K)
            sum_emb_sq = np.sum(emb_vals ** 2, axis=1)             # (B, K)
            inter = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)  # (B,)
            logits = bias[0] + lin_sum + inter
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Gradients (binary cross-entropy)
            grad_logits = (probs - yb).astype(np.float32)  # (B,)
            # bias
            g_bias = np.mean(grad_logits)
            # linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g_lin[j, v] = np.mean(grad_logits[mask])
            g_lin += L2 * linear
            # embeddings
            g_emb = np.zeros_like(embed)
            # For each feature j, gradient wrt its embedding vector for value v:
            # d(inter)/dV_{j,v} = (sum_emb - V_{j,v}) * grad_logits (broadcast over K)
            # We accumulate per (j, v)
            for j in range(d):
                vals = xb[:, j]
                V_jv = emb_vals[:, j, :]  # (B, K)
                # sum_emb is (B, K)
                diff = sum_emb - V_jv      # (B, K)
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g_emb[j, v] = np.mean(diff[mask] * grad_logits[mask, None], axis=0)
            g_emb += L2 * embed

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias[0] -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # embed
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            embed -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    # Header: magic, d, K, C (C=2 for compatibility, but we only use binary)
    hdr = struct.pack("<4i", 0x464D31, d, K, 2)
    # Flatten in order: bias (1), linear (d*256), embed (d*256*K)
    out = hdr
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary tabular classification.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d from header). Output: one byte (0 or 1). */
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
    if (hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + 4L * (1L + d * 256L + d * 256L * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(wb + 16);
    const float *linear = bias + 1;
    const float *embed = linear + d * 256L;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = 0.0f;
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_emb_sq, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += linear[j * 256 + v];
            const float *ev = embed + (j * 256L + v) * K;
            for (int f = 0; f < K; f++) {
                float val = ev[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float diff = sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
            inter += 0.5f * diff;
        }
        float logit = bias[0] + lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```