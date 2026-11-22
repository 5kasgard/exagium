```hypothesis
Family: label-fm (Factorization Machine, degree-2). The Adult dataset has strong pairwise interactions (education×occupation, marital-status×sex, etc.) that linear models miss. FM captures all pairwise feature interactions via low-rank factorization (k=8) with O(d·k) inference (d=14), far cheaper than GBDT. The byte-per-feature input maps naturally to FM's one-hot-per-value design. Failure: dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 3e-3, 1e-4, 256, 30


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    bias = np.zeros(1, dtype=np.float32)

    mw, mV, mbias = np.zeros_like(w), np.zeros_like(V), np.zeros_like(bias)
    vw, vV, vbias = np.zeros_like(w), np.zeros_like(V), np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            linear = np.zeros(B, dtype=np.float32)
            for j in range(d):
                linear += w[j, xb[:, j]]

            sum_V = np.zeros((B, K), dtype=np.float32)
            sum_V2 = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                V_j = V[j, xb[:, j], :]
                sum_V += V_j
                sum_V2 += V_j * V_j
            interaction = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)

            logits = bias[0] + linear + interaction
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)

            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            gbias = np.zeros_like(bias)

            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g = grad[mask]
                        gw[j, v] = np.sum(g)
                        diff = sum_V[mask] - V[j, v, :]
                        gV[j, v, :] = np.sum(g[:, None] * diff, axis=0)

            gbias[0] = np.sum(grad)

            gw += L2 * w
            gV += L2 * V

            t += 1
            for param, gr, m, v in [(w, gw, mw, vw), (V, gV, mV, vV), (bias, gbias, mbias, vbias)]:
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    if (d != 14 || C != 2 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)d * 256 * K + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *V = w + (long)d * 256;
    const float *bias = V + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = 0.0f;
        float sum_V[64], sum_V2[64];
        for (int kk = 0; kk < K; kk++) { sum_V[kk] = 0.0f; sum_V2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vjv[kk];
                sum_V[kk] += val;
                sum_V2[kk] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++)
            interaction += sum_V[kk] * sum_V[kk] - sum_V2[kk];
        interaction *= 0.5f;

        float logit = bias[0] + linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```