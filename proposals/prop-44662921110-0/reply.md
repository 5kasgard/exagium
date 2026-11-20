```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, capturing non-linearities that linear models miss while staying far cheaper than GBDT/MLP. The adult task has known feature interactions (e.g., education × occupation), and FMs excel on sparse tabular data. If dev accuracy ≤ label-linear (0.8524) or inference cost ≥ label-mlp16 (2325 Ir/item), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D31, d, k, C; float32 bias, linear[d][256], emb[d][256][k] (row-major)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 10


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
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = xb.shape[0]

            # Forward
            lin_part = np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
            sum_v = np.sum(emb[np.arange(d)[:, None], xb.T], axis=0)       # (bsz, K)
            sum_sq_v = np.sum(emb[np.arange(d)[:, None], xb.T] ** 2, axis=0)  # (bsz, K)
            inter = 0.5 * np.sum(sum_v ** 2 - sum_sq_v, axis=1)           # (bsz,)
            logits = bias + lin_part + inter                               # (bsz,)

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yb                                                  # (bsz,)

            # Backward
            g_bias = np.mean(grad)
            # Linear gradients
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            g_lin /= bsz
            g_lin += L2 * linear

            # Embedding gradients
            g_emb = np.zeros_like(emb)
            for j in range(d):
                v_j = emb[j, xb[:, j]]                 # (bsz, K)
                # d(inter)/dv_j = (sum_v - v_j) * grad[:, None]
                factor = (sum_v - v_j) * grad[:, None] # (bsz, K)
                np.add.at(g_emb[j], xb[:, j], factor)
            g_emb /= bsz
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Layout: int32 magic, d, K, C; float32 bias; float32 linear[d][256]; float32 emb[d][256][K]
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        memset(sum_v, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) sum_v[u] += e[u];
        }

        float sum_sq = 0.0f, sq_sum = 0.0f;
        for (int u = 0; u < K; u++) {
            sum_sq += sum_v[u] * sum_v[u];
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) sq_sum += e[u] * e[u];
        }
        score += 0.5f * (sum_sq - sq_sum);

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```