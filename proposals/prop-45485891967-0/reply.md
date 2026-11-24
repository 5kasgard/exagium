```hypothesis
Factorization Machine (label-fm) captures pairwise feature interactions crucial for tabular data like Adult, using latent vectors per feature value. Precomputed lookup tables enable O(d*K) inference (d=14, K=8), faster than MLP while modeling interactions linear models miss. Should exceed label-linear (0.8524) and approach label-mlp16 (0.8591) at lower cost. Failure if interactions don't improve over linear or if overfitting on 26k samples.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 50

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

    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    embed = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    m_lin = np.zeros_like(linear); v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed); v_emb = np.zeros_like(embed)
    m_bias = 0.0; v_bias = 0.0
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            B = len(idx)

            emb_sum = np.zeros((B, K), np.float32)
            for j in range(d):
                emb_sum += embed[j, xi[:, j]]

            s_sq = np.sum(emb_sum * emb_sum, axis=1)
            v_sq_sum = np.zeros(B, np.float32)
            for j in range(d):
                v_sq_sum += np.sum(embed[j, xi[:, j]] ** 2, axis=1)

            lin = bias + np.sum(linear[np.arange(d), xi.T], axis=0)
            logits = lin + 0.5 * (s_sq - v_sq_sum)

            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yi

            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)
            g_bias = np.sum(grad)

            for i in range(B):
                g = grad[i]
                for j in range(d):
                    v = xi[i, j]
                    g_lin[j, v] += g
                    diff = emb_sum[i] - embed[j, v]
                    g_emb[j, v] += g * diff

            g_lin += L2 * linear
            g_emb += L2 * embed

            t += 1
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4if", 0x464D4C31, d, K, C, bias)
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float bias = *(float *)(hd + 4);
    const float *linear = (const float *)(hd + 5);
    const float *embed = linear + (long)d * 256;
    long expected = 20 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = bias;
        for (int u = 0; u < K; u++) emb_sum[u] = 0.0f;
        float v_sq_sum = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lptr = linear + ((long)j * 256 + v);
            lin_sum += *lptr;
            const float *eptr = embed + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float ev = eptr[u];
                emb_sum[u] += ev;
                v_sq_sum += ev * ev;
            }
        }

        float s_sq = 0.0f;
        for (int u = 0; u < K; u++) s_sq += emb_sum[u] * emb_sum[u];
        float inter = 0.5f * (s_sq - v_sq_sum);
        float logit = lin_sum + inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```