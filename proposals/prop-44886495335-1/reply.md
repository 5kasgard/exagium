```hypothesis
The optdigits task consists of 8×8 handwritten digits with ordinal pixel counts (0–16). Linear models miss pairwise pixel interactions (e.g., strokes, corners) that are informative for digit recognition. A degree-2 polynomial logistic regression explicitly models all 2080 pairwise products on top of the 64 linear terms, adding expressive power while keeping inference to ~21k multiply-adds per item (≈33× label-linear but still microseconds). If dev accuracy does not exceed label-linear by ≥0.5% absolute, the extra compute is not justified.
```

```python
"""Polynomial degree-2 logistic regression for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, C, means[d], stds[d], bias[C], W_lin[d*C], W_quad[(d*(d+1)//2)*C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2" in little-endian


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
    rng = np.random.default_rng(SEED)

    # Ordinal z-scoring per feature
    means = X.mean(axis=0).astype(np.float32)
    stds = X.std(axis=0).astype(np.float32)
    stds[stds == 0] = 1.0
    Z = ((X.astype(np.float32) - means) / stds).astype(np.float32)  # (n, d)

    # Polynomial expansion: linear + quadratic (i<=j)
    n = Z.shape[0]
    quad_terms = [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
    Z_quad = np.column_stack(quad_terms).astype(np.float32)  # (n, d*(d+1)//2)
    Z_poly = np.hstack([Z, Z_quad]).astype(np.float32)       # (n, D)
    D = Z_poly.shape[1]

    # Initialize weights
    scale = math.sqrt(6.0 / D)
    W = (rng.uniform(-1, 1, (D, C)) * scale).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_poly[idx], y[idx]
            # Softmax gradient
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)  # (batch, C)
            # Gradients
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Split weights back to linear and quadratic parts
    W_lin = W[:d]          # (d, C)
    W_quad = W[d:]         # (d*(d+1)//2, C)

    # Pack weights file
    out = struct.pack("<3i", MAGIC, d, C)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += W_quad.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Degree-2 polynomial logistic regression for ordinal byte features.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index. */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x504F4C32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2];
    if (d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    int D_quad = d * (d + 1) / 2;
    int D = d + D_quad;
    long expected = 12L + 4L * d + 4L * d + 4L * C + 4L * d * C + 4L * D_quad * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *means = (const float *)(wb + 12);
    const float *stds  = means + d;
    const float *bias  = stds + d;
    const float *W_lin = bias + C;
    const float *W_quad = W_lin + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Normalize
        for (int j = 0; j < d; j++) z[j] = (xi[j] - means[j]) / stds[j];
        // Linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            const float *w = W_lin + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += zj * w[c];
        }
        // Quadratic part
        const float *wq = W_quad;
        for (int ii = 0; ii < d; ii++) {
            float zi = z[ii];
            for (int jj = ii; jj < d; jj++) {
                float zij = zi * z[jj];
                for (int c = 0; c < C; c++) scores[c] += zij * wq[c];
                wq += C;
            }
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```