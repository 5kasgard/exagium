```hypothesis
Family: label-fm (Factorization Machine). Adult income depends on feature interactions (e.g., education×occupation); FM captures all pairwise interactions via low-rank embeddings (K=8). Adds ~130KB weights and ~250 float ops/item (2–3× linear), targeting 0.86+ accuracy (near gbdt 0.8664) at ~1500 Ir/item. Fails if dev accuracy ≤ 0.853 or Ir/item > 2000.
```

```python
import json, math, struct, sys
import numpy as np

K = 8
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
    d = card["input"]["bytes"]; C = card["classes"]; n = X.shape[0]
    assert C == 2
    rng = np.random.default_rng(SEED)

    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.uniform(-0.01, 0.01, (d, 256, K)) * math.sqrt(6.0 / (256 * K))).astype(np.float32)

    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear); v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed);  v_emb = np.zeros_like(embed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            lin_part = np.full(B, bias, dtype=np.float32)
            for f in range(d):
                lin_part += linear[f, xb[:, f]]

            sum_v = np.zeros((B, K), dtype=np.float32)
            sum_v_sq = np.zeros((B, K), dtype=np.float32)
            for f in range(d):
                e = embed[f, xb[:, f], :]
                sum_v += e
                sum_v_sq += e * e
            inter_part = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
            logit = lin_part + inter_part

            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)

            g_bias = float(np.mean(grad_logit))
            g_lin = np.zeros_like(linear)
            for f in range(d):
                np.add.at(g_lin[f], xb[:, f], grad_logit)
            g_lin /= B; g_lin += L2 * linear

            g_emb = np.zeros_like(embed)
            for f in range(d):
                xf = xb[:, f]
                diff = sum_v - embed[f, xf]
                weighted = grad_logit[:, None] * diff
                np.add.at(g_emb[f], xf, weighted)
            g_emb /= B; g_emb += L2 * embed

            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int f = 0; f < d; f++)
            logit += linear[(long)f * 256 + xi[f]];

        for (int kk = 0; kk < K; kk++) {
            sum_v[kk] = 0.0f;
            sum_v_sq[kk] = 0.0f;
        }
        for (int f = 0; f < d; f++) {
            const float *e = embed + ((long)f * 256 + xi[f]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_v[kk] += val;
                sum_v_sq[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++)
            inter += sum_v[kk] * sum_v[kk] - sum_v_sq[kk];
        logit += 0.5f * inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    free(sum_v_sq);
    return 0;
}
```