```hypothesis
label-poly2: Degree-2 polynomial expansion of the 64 ordinal pixel features followed by L2-regularized logistic regression (Adam). Optdigits strokes create strong pairwise pixel correlations (e.g., adjacent pixels on a curve); explicit quadratic terms model this topology better than a linear model. If dev accuracy ≤ 0.9613 (label-linear), the hypothesis fails.
```
```python
"""label-poly2: Degree-2 polynomial features + logistic regression for ordinal byte inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, C, poly_dim; float32 mean[d], std[d], poly_mean[poly_dim], poly_std[poly_dim], W[poly_dim][C], b[C]."""
import json, math, struct, sys

import numpy as np

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
    C = card["classes"]
    d = X.shape[1]

    # All features are ordinal per the card
    rng = np.random.default_rng(SEED)

    # Z-score original features
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Z = (Xf - mean) / std  # (n, d)

    # Build degree-2 polynomial features: for i<=j, Z[:,i] * Z[:,j]
    # Number of features: d*(d+1)/2
    poly_dim = d * (d + 1) // 2
    P = np.empty((X.shape[0], poly_dim), dtype=np.float32)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            P[:, idx] = Z[:, i] * Z[:, j]
            idx += 1

    # Normalize polynomial features
    poly_mean = P.mean(axis=0).astype(np.float32)
    poly_std = P.std(axis=0).astype(np.float32)
    poly_std[poly_std == 0] = 1.0
    P = (P - poly_mean) / poly_std

    n, k = P.shape  # k == poly_dim
    assert k == poly_dim

    # Initialize weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            z, yy = P[idx_batch], y[idx_batch]

            # Softmax gradient
            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P_exp = np.exp(S)
            P_exp /= P_exp.sum(axis=1, keepdims=True)
            P_exp[np.arange(len(yy)), yy] -= 1.0
            g = (P_exp / len(yy)).astype(np.float32)

            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)

            t += 1
            for p, gr, (m, v) in zip(params, [grad_W, grad_b], state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    # magic, d, C, poly_dim
    out = struct.pack("<4i", 0x504F4C32, d, C, poly_dim)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += poly_mean.astype("<f4").tobytes()
    out += poly_std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 infer: ./infer weights.bin in.rec N
 * Input: d bytes (ordinal). Output: 1 byte class.
 * Weights format: magic(0x504F4C32), d, C, poly_dim; mean[d], std[d], poly_mean[poly_dim], poly_std[poly_dim], W[poly_dim][C], b[C] (all float32 LE). */
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
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x504F4C32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], poly_dim = hdr[3];
    if (d <= 0 || C <= 0 || C > 256 || poly_dim != d * (d + 1) / 2) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }

    float *ptr = (float *)(hdr + 4);
    float *mean = ptr; ptr += d;
    float *std = ptr; ptr += d;
    float *poly_mean = ptr; ptr += poly_dim;
    float *poly_std = ptr; ptr += poly_dim;
    float *W = ptr; ptr += (long)poly_dim * C;
    float *b = ptr; ptr += C;

    size_t expected = 16 + 4L * (2L * d + 2L * poly_dim + (long)poly_dim * C + C);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *p = malloc(sizeof(float) * poly_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize input features
        for (int j = 0; j < d; j++) {
            z[j] = ((float)xi[j] - mean[j]) / std[j];
        }

        // Build polynomial features (degree 2, i<=j)
        int idx = 0;
        for (int u = 0; u < d; u++) {
            for (int v = u; v < d; v++) {
                p[idx] = (z[u] * z[v] - poly_mean[idx]) / poly_std[idx];
                idx++;
            }
        }

        // Linear classification
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < poly_dim; f++) {
            float pv = p[f];
            if (pv == 0.0f) continue;
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += pv * wf[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```