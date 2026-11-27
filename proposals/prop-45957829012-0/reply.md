```hypothesis
Factorization Machine (label-fm): learns pairwise feature interactions via low-rank embeddings (K=8).
Adult income has strong interactions (education×occupation, marital×relationship) that linear models miss.
FM inference is O(d·K) using the sum-square trick, faster than MLP/GBDT.
Failure if dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ label-mlp16 (2325).
```

```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    assert card["classes"] == 2

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(idx)

            # Forward
            linear = w0 + np.sum(w[np.arange(d), x.T], axis=0).astype(np.float32)
            sum_v = np.sum(V[np.arange(d), x.T], axis=0)  # [B, K]
            sum_v2 = np.sum(V[np.arange(d), x.T] ** 2, axis=0)
            interaction = 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1).astype(np.float32)
            score = linear + interaction
            prob = sigmoid(score).astype(np.float32)

            grad = prob - yy  # [B], float32

            # Gradients
            dw0 = np.mean(grad) + L2 * w0

            dw = np.zeros_like(w)
            for f in range(d):
                counts = np.bincount(x[:, f], minlength=256).astype(np.float32)
                sums = np.bincount(x[:, f], weights=grad, minlength=256).astype(np.float32)
                dw[f] = np.where(counts > 0, sums / counts, 0) + L2 * w[f]

            dV = np.zeros_like(V)
            for f in range(d):
                for v in range(256):
                    mask = (x[:, f] == v)
                    if np.any(mask):
                        g = grad[mask][:, None]
                        sv = sum_v[mask]
                        dV[f, v] = np.mean(g * (sv - V[f, v]), axis=0) + L2 * V[f, v]

            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * dw * dw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV * dV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Save: magic, d, K, C, w0, w[d*256], V[d*256*K]
    magic = 0x464D3130
    out = struct.pack("<4if", magic, d, K, card["classes"], w0)
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
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0 || K > 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(wb + 16);
    float w0 = ptr[0];
    float *w = ptr + 1;
    float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    float *sum_v2 = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int f = 0; f < d; f++) score += w[f * 256 + xi[f]];

        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));
        for (int f = 0; f < d; f++) {
            const float *V_fv = V + ((long)f * 256 + xi[f]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = V_fv[kk];
                sum_v[kk] += v;
                sum_v2[kk] += v * v;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        score += 0.5f * inter;

        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_v);
    free(sum_v2);
    free(wb);
    if (x) free(x);
    return 0;
}
```