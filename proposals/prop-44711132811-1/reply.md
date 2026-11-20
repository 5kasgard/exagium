```hypothesis
Family: label-poly2 – logistic regression on degree-2 polynomial features (linear, squared, and all pairwise products) of z-scored 8×8 pixels.
Why: Digit shapes rely on pixel interactions (strokes, loops); pairwise terms capture this without deep nets. The 2144 features stay small enough for fast inference (~2k FLOPs/item) and should beat the linear baseline (0.9613).
Failure: If dev accuracy ≤ 0.962, the feature expansion adds no useful signal over linear.
```

```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"


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
    d = card["input"]["bytes"]          # 64
    C = card["classes"]                 # 10
    rng = np.random.default_rng(SEED)

    # per-pixel z-scoring
    mean = X.mean(axis=0).astype(np.float32)
    std  = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = (X - mean) / std                # (n, d) float64

    # degree-2 polynomial features: linear, squared, pairwise
    n = Z.shape[0]
    num_feat = d + d + d * (d - 1) // 2  # 64 + 64 + 2016 = 2144
    Phi = np.zeros((n, num_feat), dtype=np.float32)
    Phi[:, :d] = Z
    Phi[:, d:2*d] = Z**2
    idx = 2 * d
    for i in range(d):
        for j in range(i + 1, d):
            Phi[:, idx] = Z[:, i] * Z[:, j]
            idx += 1

    # logistic regression with Adam
    k = num_feat
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            phi = Phi[idx_batch]
            yy = y[idx_batch]
            S = phi @ W + b
            g = softmax_grad(S, yy)
            grad_W = phi.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # save weights
    out = struct.pack("<4i", MAGIC, d, num_feat, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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

#define MAGIC 0x504F4C32  /* "POL2" */

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
    if (nw < 16 || hd[0] != MAGIC) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_feat = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * num_feat * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *mean = (float *)(hd + 4);
    float *std  = mean + d;
    float *W    = std + d;
    float *b    = W + (long)num_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            float s = std[j];
            z[j] = (s != 0.0f) ? (xi[j] - mean[j]) / s : 0.0f;
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        /* linear terms */
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * W[(long)j * C + c];
        }
        /* squared terms */
        for (int j = 0; j < d; j++) {
            float zj2 = z[j] * z[j];
            for (int c = 0; c < C; c++) scores[c] += zj2 * W[(long)(d + j) * C + c];
        }
        /* pairwise interactions */
        long idx = 2L * d;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1 + 1; i2 < d; i2++) {
                float prod = z[i1] * z[i2];
                for (int c = 0; c < C; c++) scores[c] += prod * W[idx * C + c];
                idx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }

    free(z); free(scores);
    return 0;
}
```