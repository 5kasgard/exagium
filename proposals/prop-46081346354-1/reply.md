```hypothesis
Family: label-poly2 (degree-2 polynomial features + linear classifier).
Why: Pixel interactions (e.g., strokes) matter for 8x8 digits; degree-2 terms capture them with only 2145 features.
Failure: Dev accuracy ≤ 0.9613 (label-linear) or latency ≥ 776 ns/item (label-mlp16).
```

```python
"""label-poly2: polynomial features (degree 2) + logistic regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x504F4C32, d, C, num_features; float32 mean[d], std[d], W[num_features*C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def poly2_expand(Z):  # Z: (n, d) float32 z-scored
    n, d = Z.shape
    # features: 1 (bias) + d linear + d*(d+1)/2 quadratic
    nf = 1 + d + d * (d + 1) // 2
    Phi = np.empty((n, nf), dtype=np.float32)
    Phi[:, 0] = 1.0
    Phi[:, 1:1 + d] = Z
    col = 1 + d
    for j in range(d):
        for k in range(j, d):
            Phi[:, col] = Z[:, j] * Z[:, k]
            col += 1
    return Phi


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # z-score per pixel on full training set
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    Phi = poly2_expand(Z)
    n, nf = Phi.shape

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (nf, C)) * math.sqrt(6.0 / nf)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Phi[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights
    out = struct.pack("<4i", MAGIC, d, C, nf)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class 0..9.
 * Weights: magic 0x504F4C32, d, C, nf; float32 mean[d], std[d], W[nf*C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], nf = hd[3];
    if (C < 1 || C > 256 || d != 64 || nf != 1 + d + d * (d + 1) / 2) {
        fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * nf * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)nf * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * nf);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++)
            z[j] = (xi[j] - mean[j]) / std[j];

        phi[0] = 1.0f;
        memcpy(phi + 1, z, d * sizeof(float));
        int col = 1 + d;
        for (int j = 0; j < d; j++)
            for (int k2 = j; k2 < d; k2++)
                phi[col++] = z[j] * z[k2];

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < nf; f++) {
            float ph = phi[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += ph * wf[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```