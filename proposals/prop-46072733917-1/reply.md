```hypothesis
Family: label-poly2 (degree-2 polynomial features + linear). The 8x8 digit pixels have strong pairwise interactions (strokes, corners) that a linear model misses. A degree-2 expansion (64 linear + 2080 pairwise terms) captures these with a single linear layer, keeping inference fast (~23k FMAs/item). If dev accuracy ≤ 0.965, the expansion doesn't justify its cost over label-linear.
```
```python
"""label-poly2: Degree-2 polynomial features + linear classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, C, nf; float32 mu[d], inv_sd[d], W[nf][C], b[C].
nf = d + d*(d+1)//2 = 64 + 2080 = 2144 for d=64."""
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


def poly_features(Z):
    """Z: (n, d) float32 z-scored. Returns (n, nf) with linear + pairwise (i<=j)."""
    n, d = Z.shape
    nf = d + d * (d + 1) // 2
    F = np.empty((n, nf), dtype=np.float32)
    F[:, :d] = Z
    col = d
    for i in range(d):
        zi = Z[:, i:i+1]
        F[:, col:col + d - i] = zi * Z[:, i:]
        col += d - i
    return F


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Z-score per pixel on full training set
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # Polynomial features
    F = poly_features(Z)
    nf = F.shape[1]

    # Split last 20% for tuning (per card), but we use fixed hyperparams; train on all
    rng = np.random.default_rng(SEED)

    # Linear weights
    W = np.zeros((nf, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            f, yy = F[idx], y[idx]
            g = softmax_grad(f @ W + b, yy)
            grad_W = f.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    inv_sd = (1.0 / sd).astype(np.float32)
    mu32 = mu.astype(np.float32)
    magic = 0x504F4C32  # "POL2"
    hdr = struct.pack("<4i", magic, d, C, nf)
    out = hdr + mu32.tobytes() + inv_sd.tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input: d bytes (pixels). Output: 1 byte class (argmax).
 * Weights format: magic(0x504F4C32), d, C, nf; float32 mu[d], inv_sd[d], W[nf][C], b[C].
 * nf = d + d*(d+1)/2. Features: z[i] = (x[i]-mu[i])*inv_sd[i]; linear z[0..d-1]; pairwise z[i]*z[j] for i<=j. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], nf = hd[3];
    if (d != 64 || C < 1 || C > 256 || nf != d + d*(d+1)/2) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L*d + 4L*d + 4L*(long)nf*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *inv_sd = mu + d;
    const float *W = inv_sd + d;
    const float *b = W + (long)nf * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *f = malloc(sizeof(float) * nf);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) * inv_sd[j];

        // linear features
        memcpy(f, z, d * sizeof(float));
        int idx = d;
        // pairwise i<=j
        for (int i1 = 0; i1 < d; i1++) {
            float zi = z[i1];
            for (int j1 = i1; j1 < d; j1++) {
                f[idx++] = zi * z[j1];
            }
        }

        // scores = f @ W + b
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int k1 = 0; k1 < nf; k1++) {
            float fk = f[k1];
            const float *wk = W + (long)k1 * C;
            for (int c = 0; c < C; c++) sc[c] += fk * wk[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```