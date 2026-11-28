```hypothesis
Family: label-poly2 (degree-2 polynomial expansion + logistic regression).
Why: Optdigits pixels have strong local correlations; pairwise products capture stroke intersections and curvatures that linear models miss. The explicit polynomial kernel is known to perform well on this dataset (often >98% with SVM).
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds 2× label-linear without accuracy gain.
```

```python
"""label-poly2: degree-2 polynomial expansion + logistic regression for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Exports: magic, d, C, feat_dim, mean[d], std[d], W[feat_dim*C]."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Standardize each pixel (ordinal)
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Zlin = (Xf - mean) / std  # (n, d)

    # Build polynomial features: bias, linear, quadratic (upper triangle)
    # feat_dim = 1 + d + d*(d+1)//2
    feat_dim = 1 + d + d * (d + 1) // 2
    Z = np.empty((n, feat_dim), dtype=np.float32)
    Z[:, 0] = 1.0  # bias
    Z[:, 1:1 + d] = Zlin
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            Z[:, col] = Zlin[:, i] * Zlin[:, j]
            col += 1

    rng = np.random.default_rng(SEED)
    # Xavier init for W (feat_dim, C)
    scale = math.sqrt(6.0 / feat_dim)
    W = (rng.uniform(-1, 1, (feat_dim, C)) * scale).astype(np.float32)
    params = [W]
    state = [(np.zeros_like(W), np.zeros_like(W))]
    t = 0

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W
            g = softmax_grad(logits, yy)
            grad = z.T @ g + L2 * W
            t += 1
            m, v = state[0]
            m *= 0.9; m += 0.1 * grad
            v *= 0.999; v += 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, C, feat_dim)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()  # row-major: feat_dim x C
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input: d bytes (ordinal 0..16). Output: one byte class (0..9).
 * Weights format: magic(0x504F4C32), d, C, feat_dim, mean[d], std[d], W[feat_dim*C] (row-major float32). */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], feat_dim = hd[3];
    if (d != 64 || C != 10 || feat_dim != 1 + d + d * (d + 1) / 2) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    long expected = 16 + 4L * d * 2 + 4L * feat_dim * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W = std + d;  // [feat_dim][C] row-major

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // standardize
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];

        // bias
        for (int c = 0; c < C; c++) logits[c] = W[c];  // W[0][c]

        // linear terms
        for (int j = 0; j < d; j++) {
            const float *w = W + (1 + j) * C;
            float zj = z[j];
            for (int c = 0; c < C; c++) logits[c] += zj * w[c];
        }

        // quadratic terms (upper triangle)
        int idx = 1 + d;
        for (int j = 0; j < d; j++) {
            for (int k = j; k < d; k++) {
                const float *w = W + idx * C;
                float zjk = z[j] * z[k];
                for (int c = 0; c < C; c++) logits[c] += zjk * w[c];
                idx++;
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```