```hypothesis
Family: label-quadratic (quadratic logistic regression on z-scored pixels).
Why: Pixel interactions (strokes, curves) matter for digits; a quadratic model captures pairwise
     products while staying convex and fast. 64 features → 2145 terms (1 + 64 + 64*65/2).
     Training with Adam (same recipe as label-linear) should beat 0.9613; inference ~21k FLOPs/item.
Failure signal: dev accuracy ≤ 0.9613 or instructions/item > 20k (≈4× label-linear).
```

```python
"""Quadratic logistic regression for fixed-size byte items (ordinal only).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x51554144, int32 d C, float32 mean[d], std[d],
                float32 W[C][1 + d + d*(d+1)/2] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x51554144  # "QUAD"


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score each pixel (all ordinal per card)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)          # (n, d)

    # polynomial features: [1, z_1 ... z_d, z_1^2, z_1 z_2, ..., z_d^2]
    n = Z.shape[0]
    nf = 1 + d + d * (d + 1) // 2
    Phi = np.empty((n, nf), dtype=np.float32)
    Phi[:, 0] = 1.0
    Phi[:, 1:1+d] = Z
    col = 1 + d
    for j in range(d):
        Phi[:, col:col + d - j] = Z[:, j:j+1] * Z[:, j:]
        col += d - j

    # softmax regression with Adam
    W = (rng.uniform(-1, 1, (C, nf)) * math.sqrt(6.0 / nf)).astype(np.float32)
    m = np.zeros_like(W); v = np.zeros_like(W)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]
            logits = phi @ W.T                         # (b, C)
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs.T @ phi) / len(yy) + L2 * W  # (C, nf)
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<3i", MAGIC, d, C))
        f.write(mu.astype("<f4").tobytes())
        f.write(sd.astype("<f4").tobytes())
        f.write(W.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* Quadratic logistic regression: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights: magic 0x51554144, int32 d C, float32 mean[d], std[d], W[C][1+d+d*(d+1)/2]. */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 12 || hdr[0] != 0x51554144) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], C = hdr[2];
    int nf = 1 + d + d * (d + 1) / 2;
    long expected = 12L + 4L * d * 2 + 4L * C * nf;
    if (nw != expected || C < 1 || C > 256) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hdr + 3);
    const float *std  = mean + d;
    const float *W    = std + d;                // [C][nf]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = malloc(sizeof(float) * nf);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score
        for (int j = 0; j < d; j++) {
            float z = (xi[j] - mean[j]) / std[j];
            phi[1 + j] = z;
        }
        phi[0] = 1.0f;

        // quadratic terms
        int idx = 1 + d;
        for (int j = 0; j < d; j++) {
            for (int k2 = j; k2 < d; k2++) {
                phi[idx++] = phi[1 + j] * phi[1 + k2];
            }
        }

        // scores = W * phi
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *wc = W + (long)c * nf;
            for (int f = 0; f < nf; f++) sum += wc[f] * phi[f];
            scores[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```