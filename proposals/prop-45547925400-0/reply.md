```hypothesis
Family: label-fm (Factorization Machine with k=8 factors). All 14 features treated as 256-value categorical; learns per-value linear weight and k-dim embedding. Captures all pairwise interactions in O(d*k) time. Should beat label-linear (0.8524) on accuracy while keeping inference cost ~2x linear (~1500 Ir/item). Fails if dev accuracy <= 0.8524 or Ir/item > 2000.
```
```python
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    bias = np.float32(0.0)
    linear = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    embed = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)

    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            lin_sum = np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)
            emb_vals = embed[np.arange(d)[:, None], xb.T]
            sum_v = np.sum(emb_vals, axis=0)
            sum_v2 = np.sum(emb_vals * emb_vals, axis=0)
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            logit = bias + lin_sum + interaction

            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32) / bsz

            g_bias = np.sum(grad_logit)
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad_logit)
            g_lin += L2 * linear

            g_emb = np.zeros_like(embed)
            for j in range(d):
                vals = xb[:, j]
                diff = sum_v - embed[j, vals]
                g_emb_j = diff * grad_logit[:, None]
                np.add.at(g_emb[j], vals, g_emb_j)
            g_emb += L2 * embed

            t += 1
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (math.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)

            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)

            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            embed -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    out = struct.pack("<4i", 0x464D3100, d, K, C)
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }
        for (int f = 0; f < K; f++) {
            sum_v[f] = 0.0f;
            sum_v2[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *v = embed + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float val = v[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }

        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```