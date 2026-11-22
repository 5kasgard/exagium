```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Optdigits pixels have strong pairwise interactions (strokes, loops) that a linear model misses. Adding all 2080 quadratic terms to the 64 normalized pixels gives a rich feature space while keeping the model a simple linear classifier. Training uses the same deterministic Adam as label-linear; inference requires ~2145 multiply-adds per class (10 classes) — still far cheaper than GBDT or KNN.
Failure signal: If dev accuracy ≤ 0.965 (not clearly above label-linear's 0.9613) the extra complexity is not justified.
```

```python
"""label-poly2: polynomial degree-2 logistic regression for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C59, d, C, D; float32 means[d], stds[d], weights[C][D] (row-major, bias included as first feature)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C59  # "POLY"


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

    # z-score each ordinal feature
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)  # (n, d)

    # Build polynomial features: [1, z_i, z_i*z_j (i<=j)]
    n = Z.shape[0]
    D = 1 + d + d * (d + 1) // 2
    Phi = np.empty((n, D), dtype=np.float32)
    Phi[:, 0] = 1.0
    Phi[:, 1:1+d] = Z
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = Z[:, i] * Z[:, j]
            col += 1

    # Initialize weights (C, D)
    W = (rng.uniform(-1, 1, (C, D)) * math.sqrt(6.0 / D)).astype(np.float32)
    params = [W]
    state = [(np.zeros_like(W), np.zeros_like(W))]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]
            logits = phi @ W.T
            g = softmax_grad(logits, yy)
            grad = phi.T @ g + L2 * W  # (D, C) -> transpose to (C, D)
            grad = grad.T
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    out = struct.pack("<4i", MAGIC, d, C, D)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 infer: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format: magic(0x504F4C59), d, C, D; float32 means[d], stds[d], weights[C][D] (row-major).
 * Feature vector: phi[0]=1, phi[1..d]=z_i, phi[1+d ..]=z_i*z_j (i<=j). */
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
    if (nw < 16 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d != 64 || C != 10 || D != 1 + 64 + 64*65/2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)C * D);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *means = (const float *)(hd + 4);
    const float *stds  = means + d;
    const float *W     = stds + d;  // size C * D

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            float sd = stds[j];
            z[j] = sd == 0.0f ? 0.0f : (xi[j] - means[j]) / sd;
        }

        // Compute scores = W * phi
        for (int c = 0; c < C; c++) scores[c] = W[c * D];  // bias (phi[0]=1)

        // Linear terms
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * W[c * D + 1 + j];
        }

        // Quadratic terms (i<=j)
        int base = 1 + d;
        for (int ii = 0; ii < d; ii++) {
            float zi = z[ii];
            for (int jj = ii; jj < d; jj++) {
                float prod = zi * z[jj];
                int idx = base + ii * d - ii * (ii - 1) / 2 + (jj - ii);
                for (int c = 0; c < C; c++) scores[c] += prod * W[c * D + idx];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```