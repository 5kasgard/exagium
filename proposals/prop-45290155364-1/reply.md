```hypothesis
Family: label-rff (Random Fourier Features + linear). The optdigits pixels have smooth spatial structure; an RBF kernel captures this well but is slow. RFF approximates the RBF kernel with explicit randomized features, yielding a linear model in the lifted space that is fast at inference (O(D·d) with D≈256) and deterministic with a fixed seed. It should beat label-linear (0.9613) by modeling pixel interactions. Failure: dev accuracy ≤ 0.9613 or instructions/item > 2× label-linear.
```
```python
"""label-rff: Random Fourier Features (RBF kernel approx) + linear for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal bytes by z-scoring. Generates D=256 RFF
with gamma=0.01 (w ~ N(0, gamma*I), b ~ U(0, 2pi)). Trains linear on RFF with Adam
(lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))). Exports:
  int32 magic=0x52464631, d, D, C
  float32 mean[d], std[d]
  float32 W_rff[D*d], b_rff[D]
  float32 W_out[D*C], b_out[C]"""
import json, math, struct, sys

import numpy as np

D = 256
GAMMA = 0.01
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
    rng = np.random.default_rng(SEED)

    # z-score ordinal features
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Xz = ((X.astype(np.float32) - mean) / std).astype(np.float32)

    # RFF parameters
    W_rff = (rng.normal(0, math.sqrt(GAMMA), (D, d))).astype(np.float32)
    b_rff = (rng.uniform(0, 2 * math.pi, D)).astype(np.float32)

    # Compute RFF features: sqrt(2/D) * cos(Xz @ W_rff.T + b_rff)
    Z = np.cos(Xz @ W_rff.T + b_rff) * math.sqrt(2.0 / D)

    n, k = Z.shape  # k == D
    # Linear classifier on RFF features
    W_out = np.zeros((k, C), np.float32)
    b_out = np.zeros(C, np.float32)
    params = [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = sg(z @ W_out + b_out, yy)
            grads = [z.T @ g + L2 * W_out, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x52464631, d, D, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_rff.astype("<f4").tobytes()
    out += b_rff.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (z-scored), output: 1 byte class.
 * Weights: magic, d, D, C; mean[d], std[d], W_rff[D*d], b_rff[D], W_out[D*C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || D < 1 || D > 4096 || d < 1 || d > 1024) {
        fprintf(stderr, "bad dims\n"); return 3;
    }
    long expected = 16L + 4L * (2L * d + (long)D * d + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W_rff = std + d;
    const float *b_rff = W_rff + (long)D * d;
    const float *W_out = b_rff + D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *sc = malloc(sizeof(float) * C);
    float scale = sqrtf(2.0f / D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score and compute RFF: z = cos((x-mean)/std @ W_rff.T + b_rff) * scale
        for (int j = 0; j < D; j++) {
            float sum = b_rff[j];
            const float *wrow = W_rff + (long)j * d;
            for (int f = 0; f < d; f++) {
                sum += ((float)xi[f] - mean[f]) / std[f] * wrow[f];
            }
            z[j] = cosf(sum) * scale;
        }

        // Linear output
        for (int c = 0; c < C; c++) sc[c] = b_out[c];
        for (int j = 0; j < D; j++) {
            float zj = z[j];
            const float *wrow = W_out + (long)j * C;
            for (int c = 0; c < C; c++) sc[c] += zj * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```