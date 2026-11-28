```hypothesis
Factorization Machines (FM) model all pairwise feature interactions via low-rank factorization, which is crucial for tabular data like Adult where interactions (education×occupation, marital-status×relationship) are known to be predictive. With d=14 features and factor dimension K=8, inference requires only ~112 multiply-adds per item (O(d·K)), comparable to linear models. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis that pairwise interactions matter here fails.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
SEED = 1
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

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
    d = card["input"]["bytes"]
    C = card["classes"]
    y_bin = (y * 2 - 1).astype(np.float32)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(2.0 / (256 + K))).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y_bin[idx]
            B = len(idx)

            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)
            V_gathered = V[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_V = np.sum(V_gathered, axis=1)
            sum_V_sq = np.sum(V_gathered ** 2, axis=1)
            inter = 0.5 * np.sum(sum_V ** 2 - sum_V_sq, axis=1)
            score = lin + inter

            y_score = yb * score
            sig = 1.0 / (1.0 + np.exp(y_score))
            grad_score = -yb * sig

            g_w0 = np.mean(grad_score) + L2 * w0

            g_w = np.zeros_like(w)
            for j in range(d):
                np.add.at(g_w[j], xb[:, j], grad_score)
            g_w /= B
            g_w += L2 * w

            g_V = np.zeros_like(V)
            for j in range(d):
                diff = sum_V - V_gathered[:, j, :]
                for b in range(B):
                    v = xb[b, j]
                    g_V[j, v] += grad_score[b] * diff[b]
            g_V /= B
            g_V += L2 * V

            t += 1
            m_w0 = BETA1 * m_w0 + (1 - BETA1) * g_w0
            v_w0 = BETA2 * v_w0 + (1 - BETA2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - BETA1 ** t)) / (np.sqrt(v_w0 / (1 - BETA2 ** t)) + EPS)

            m_w = BETA1 * m_w + (1 - BETA1) * g_w
            v_w = BETA2 * v_w + (1 - BETA2) * g_w * g_w
            w -= LR * (m_w / (1 - BETA1 ** t)) / (np.sqrt(v_w / (1 - BETA2 ** t)) + EPS)

            m_V = BETA1 * m_V + (1 - BETA1) * g_V
            v_V = BETA2 * v_V + (1 - BETA2) * g_V * g_V
            V -= LR * (m_V / (1 - BETA1 ** t)) / (np.sqrt(v_V / (1 - BETA2 ** t)) + EPS)

    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_buf[1];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) {
            score += w[j * 256 + xi[j]];
        }
        for (int f = 0; f < K; f++) {
            float sum_V = 0.0f, sum_V_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + f];
                sum_V += v;
                sum_V_sq += v * v;
            }
            score += 0.5f * (sum_V * sum_V - sum_V_sq);
        }

        out_buf[0] = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, out_buf, 1);
    }
    return 0;
}
```