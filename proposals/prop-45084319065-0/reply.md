```hypothesis
Family: label-fm (factorization machine with binary cross-entropy).
Why: Adult has mixed ordinal/categorical features; FM captures pairwise interactions via low-rank embeddings, improving over linear while staying fast (O(d*k) inference). The 14-byte items make embedding lookups cache-friendly.
Failure: If dev accuracy <= 0.8524 (linear) or instructions/item > 1000, the model doesn't meet the goal.
```

```python
import json, math, struct, sys
import numpy as np

K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            lin_sum = np.zeros(B, dtype=np.float32)
            for j in range(d):
                lin_sum += linear[j, xb[:, j]]

            embs = np.stack([emb[j, xb[:, j]] for j in range(d)], axis=0)
            sum_emb = embs.sum(axis=0)
            sum_emb_sq = (embs**2).sum(axis=0)
            inter = 0.5 * np.sum(sum_emb**2 - sum_emb_sq, axis=1)

            score = bias + lin_sum + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - yb).astype(np.float32)

            g_bias = grad_score.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad_score)
                diff = (sum_emb - embs[j]) * grad_score[:, None]
                np.add.at(g_emb[j], vals, diff)

            g_lin += L2 * linear
            g_emb += L2 * emb

            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)

    magic = 0x464D4249
    C = 2
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += struct.pack("<f", bias)
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4249) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(wb + 16);
    const float *lin = bias + 1;
    const float *emb = lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[32];
    float sum_emb_sq[32];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = *bias;
        for (int l = 0; l < K; l++) { sum_emb[l] = 0.0f; sum_emb_sq[l] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += lin[(long)j * 256 + val];
            const float *e = emb + ((long)j * 256 + val) * K;
            for (int l = 0; l < K; l++) {
                float ev = e[l];
                sum_emb[l] += ev;
                sum_emb_sq[l] += ev * ev;
            }
        }
        float inter = 0.0f;
        for (int l = 0; l < K; l++) {
            inter += 0.5f * (sum_emb[l] * sum_emb[l] - sum_emb_sq[l]);
        }
        score += inter;
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```