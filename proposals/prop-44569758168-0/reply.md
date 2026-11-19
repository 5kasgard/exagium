```hypothesis
Family: label-fm (factorization machine with per-feature-value embeddings).
Why: Adult has 14 mixed ordinal/categorical features; FM captures pairwise interactions with low-rank embeddings (K=8), giving richer representation than linear models while staying lightweight. The 14 features × 256 values × 8 dims ≈ 29k embedding parameters plus linear terms, small enough for fast inference.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or CPU cost exceeds label-mlp16 (≈2300 Ir/item), the model does not justify its complexity.
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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    m_bias = np.float32(0.0)
    v_bias = np.float32(0.0)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            bs = len(yy)

            linear_sum = np.sum(linear[np.arange(d), x.T], axis=0)
            s_vec = np.zeros((bs, K), dtype=np.float32)
            for i in range(d):
                s_vec += emb[i, x[:, i], :]
            s2_vec = np.zeros((bs, K), dtype=np.float32)
            for i in range(d):
                v = emb[i, x[:, i], :]
                s2_vec += v * v
            interaction = 0.5 * np.sum(s_vec * s_vec - s2_vec, axis=1)
            logit = bias + linear_sum + interaction

            p = 1.0 / (1.0 + np.exp(-logit))
            g = (p - yy).astype(np.float32)

            grad_bias = np.mean(g)

            grad_linear = np.zeros_like(linear)
            for i in range(d):
                np.add.at(grad_linear[i], x[:, i], g)
            grad_linear /= bs
            grad_linear += L2 * linear

            grad_emb = np.zeros_like(emb)
            for i in range(d):
                v_i = emb[i, x[:, i], :]
                gi = g[:, None] * (s_vec - v_i)
                for k in range(K):
                    np.add.at(grad_emb[i, :, k], x[:, i], gi[:, k])
            grad_emb /= bs
            grad_emb += L2 * emb

            t += 1
            m_bias = 0.9 * m_bias + 0.1 * grad_bias
            v_bias = 0.999 * v_bias + 0.001 * grad_bias * grad_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)

            m_linear = 0.9 * m_linear + 0.1 * grad_linear
            v_linear = 0.999 * v_linear + 0.001 * grad_linear * grad_linear
            linear -= LR * (m_linear / (1 - 0.9 ** t)) / (np.sqrt(v_linear / (1 - 0.999 ** t)) + 1e-8)

            m_emb = 0.9 * m_emb + 0.1 * grad_emb
            v_emb = 0.999 * v_emb + 0.001 * grad_emb * grad_emb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x314C4D46  # 'FML1' little-endian
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x314C4D46) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K <= 0 || C != 2) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        for (int j = 0; j < d; j++) logit += linear[(long)j * 256 + xi[j]];

        for (int kk = 0; kk < K; kk++) { sum_emb[kk] = 0.0f; sum_emb_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = emb + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = v[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_emb[kk] * sum_emb[kk] - sum_emb_sq[kk];
        logit += 0.5f * inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```