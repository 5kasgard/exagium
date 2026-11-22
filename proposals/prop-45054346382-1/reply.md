```hypothesis
Family: label-rff (Random Fourier Features + linear readout). The 8x8 optdigits pixels have non-linear structure that a linear model misses; a fixed random Fourier projection (64→64) gives a cheap non-linear lift while keeping inference a single matrix multiply. We expect >0.965 dev accuracy (beating label-linear's 0.9613) at ~1.5× the instruction cost. If dev accuracy ≤0.962 or cost >2× label-linear, the family is not worthwhile.
```

```python
"""label-rff: Random Fourier Features + linear softmax for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single-thread, numpy+stdlib only.
Exports: magic 0x52464631, d, D, C; float32 mean[d], std[d], W_rand[D][d], b_rand[D], W_lin[D][C], b_lin[C]."""
import json, math, struct, sys

import numpy as np

D = 64                 # number of random Fourier features
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 42, 3e-3, 1e-4, 64, 600000, 100


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

    # per-feature z-score (ordinal only in this task)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Xz = ((X - mean) / std).astype(np.float32)

    # fixed random projection for RFF: W_rand [D, d], b_rand [D]
    W_rand = rng.normal(0, 1, size=(D, d)).astype(np.float32)
    b_rand = rng.uniform(0, 2 * math.pi, size=D).astype(np.float32)

    # compute random Fourier features: sqrt(2/D) * cos(Xz @ W_rand.T + b_rand)
    Z = np.cos(Xz @ W_rand.T + b_rand) * math.sqrt(2.0 / D)   # [n, D]

    n, k = Z.shape
    # linear softmax on top
    W_lin = np.zeros((k, C), np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W_lin + b_lin
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W_lin, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights
    # magic, d, D, C
    out = struct.pack("<4i", 0x52464631, d, D, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_rand.astype("<f4").tobytes()
    out += b_rand.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal). Output: one byte class (argmax).
 * Weights layout (little-endian):
 *   int32 magic=0x52464631, d, D, C
 *   float32 mean[d], std[d]
 *   float32 W_rand[D][d] (row-major)
 *   float32 b_rand[D]
 *   float32 W_lin[D][C] (row-major)
 *   float32 b_lin[C] */
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
    if (nw < 16 || hdr[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], D = hdr[2], C = hdr[3];
    if (d != 64 || D != 64 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 4                       // mean
        + (size_t)d * 4                       // std
        + (size_t)D * d * 4                   // W_rand
        + (size_t)D * 4                       // b_rand
        + (size_t)D * C * 4                   // W_lin
        + (size_t)C * 4;                      // b_lin
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std  = mean + d;
    float *W_rand = std + d;
    float *b_rand = W_rand + (size_t)D * d;
    float *W_lin = b_rand + D;
    float *b_lin = W_lin + (size_t)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score + random Fourier features
        for (int u = 0; u < D; u++) {
            float acc = b_rand[u];
            const float *Wr = W_rand + (size_t)u * d;
            for (int j = 0; j < d; j++) {
                float xj = (xi[j] - mean[j]) / std[j];
                acc += xj * Wr[j];
            }
            z[u] = cosf(acc) * 0.1767766953f; // sqrt(2/64) = 1/8 = 0.125? Wait: sqrt(2/64)=sqrt(1/32)=0.1767767
        }

        // linear readout
        for (int c = 0; c < C; c++) s[c] = b_lin[c];
        for (int u = 0; u < D; u++) {
            float zu = z[u];
            const float *Wlc = W_lin + (size_t)u * C;
            for (int c = 0; c < C; c++) s[c] += zu * Wlc[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```