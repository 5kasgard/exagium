```hypothesis
Family: label-polyadj (linear model with adjacent pixel interactions).
Why: Optdigits pixels are spatially ordered; adjacent products capture local edge/curve information that a purely linear model misses, potentially boosting accuracy above 96.1% with only ~2.75x more features.
Failure: If accuracy on dev does not exceed label-linear's 0.9613, or if inference cost exceeds label-mlp16's ~8700 Ir/item.
```

```python
"""label-polyadj: linear model on original pixels plus adjacent pairwise products.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
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
    d = X.shape[1]          # 64
    assert d == 64

    # Build adjacency for 8x8 grid (horizontal and vertical neighbours)
    pairs = []
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if c + 1 < 8:
                pairs.append((idx, idx + 1))
            if r + 1 < 8:
                pairs.append((idx, idx + 8))
    n_pairs = len(pairs)    # 112
    n_feat = d + n_pairs    # 176

    # Expanded feature matrix: raw pixels + raw products
    Z_raw = np.empty((X.shape[0], n_feat), dtype=np.float32)
    Z_raw[:, :d] = X.astype(np.float32)
    for p, (i, j) in enumerate(pairs):
        Z_raw[:, d + p] = X[:, i].astype(np.float32) * X[:, j].astype(np.float32)

    # Normalize each feature
    mean = Z_raw.mean(axis=0)
    std = Z_raw.std(axis=0)
    std[std == 0] = 1.0
    Z = (Z_raw - mean) / std

    n, k = Z.shape
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights: magic, version, d, n_feat, C, mean, std, W, b
    magic = 0x504F4C59  # "POLY"
    version = 1
    header = struct.pack("<5i", magic, version, d, n_feat, C)
    body = mean.astype("<f4").tobytes() + std.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-polyadj inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 ordinal pixels). Output: one byte class.
 * Weights format: magic(4), version(4), d(4), n_feat(4), C(4),
 *   mean[n_feat](f4), std[n_feat](f4), W[n_feat*C](f4), b[C](f4). */
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

/* Hardcoded adjacency for 8x8 grid (horizontal then vertical) */
static const uint8_t adj_i[112] = {
    0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7,
    8,9, 9,10, 10,11, 11,12, 12,13, 13,14, 14,15,
    16,17, 17,18, 18,19, 19,20, 20,21, 21,22, 22,23,
    24,25, 25,26, 26,27, 27,28, 28,29, 29,30, 30,31,
    32,33, 33,34, 34,35, 35,36, 36,37, 37,38, 38,39,
    40,41, 41,42, 42,43, 43,44, 44,45, 45,46, 46,47,
    48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55,
    56,57, 57,58, 58,59, 59,60, 60,61, 61,62, 62,63,
    0,8, 1,9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15,
    8,16, 9,17, 10,18, 11,19, 12,20, 13,21, 14,22, 15,23,
    16,24, 17,25, 18,26, 19,27, 20,28, 21,29, 22,30, 23,31,
    24,32, 25,33, 26,34, 27,35, 28,36, 29,37, 30,38, 31,39,
    32,40, 33,41, 34,42, 35,43, 36,44, 37,45, 38,46, 39,47,
    40,48, 41,49, 42,50, 43,51, 44,52, 45,53, 46,54, 47,55,
    48,56, 49,57, 50,58, 51,59, 52,60, 53,61, 54,62, 55,63
};
static const uint8_t adj_j[112] = {
    1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8,
    9,10, 10,11, 11,12, 12,13, 13,14, 14,15, 15,16,
    17,18, 18,19, 19,20, 20,21, 21,22, 22,23, 23,24,
    25,26, 26,27, 27,28, 28,29, 29,30, 30,31, 31,32,
    33,34, 34,35, 35,36, 36,37, 37,38, 38,39, 39,40,
    41,42, 42,43, 43,44, 44,45, 45,46, 46,47, 47,48,
    49,50, 50,51, 51,52, 52,53, 53,54, 54,55, 55,56,
    57,58, 58,59, 59,60, 60,61, 61,62, 62,63, 63,64,
    8,16, 9,17, 10,18, 11,19, 12,20, 13,21, 14,22, 15,23,
    16,24, 17,25, 18,26, 19,27, 20,28, 21,29, 22,30, 23,31,
    24,32, 25,33, 26,34, 27,35, 28,36, 29,37, 30,38, 31,39,
    32,40, 33,41, 34,42, 35,43, 36,44, 37,45, 38,46, 39,47,
    40,48, 41,49, 42,50, 43,51, 44,52, 45,53, 46,54, 47,55,
    48,56, 49,57, 50,58, 51,59, 52,60, 53,61, 54,62, 55,63
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int version = hd[1];
    int d = hd[2];
    int n_feat = hd[3];
    int C = hd[4];
    if (d != 64 || n_feat != 176 || C != 10 || version != 1) {
        fprintf(stderr, "bad weights header: d=%d n_feat=%d C=%d ver=%d\n", d, n_feat, C, version);
        return 3;
    }
    long expected = 20L + 8L * n_feat + 4L * C * (n_feat + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 5);
    const float *std = mean + n_feat;
    const float *W = std + n_feat;
    const float *b = W + (long)n_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * n_feat);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* original pixels normalized */
        for (int f = 0; f < d; f++) {
            feat[f] = (xi[f] - mean[f]) / std[f];
        }
        /* adjacent products normalized */
        for (int p = 0; p < 112; p++) {
            float v = feat[adj_i[p]] * feat[adj_j[p]];
            feat[d + p] = (v - mean[d + p]) / std[d + p];
        }

        /* linear classification */
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int f = 0; f < n_feat; f++) {
            float w = feat[f];
            if (w == 0.0f) continue;
            const float *wrow = W + (long)f * C;
            for (int c = 0; c < C; c++) score[c] += w * wrow[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```