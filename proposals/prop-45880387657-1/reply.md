```hypothesis
Family: label-spatial. Augment the 64 raw pixels with 32 spatial aggregates (8 row sums, 8 column sums, 16 non-overlapping 2x2 block sums) → 96 features. Train a multinomial logistic regression on these features. Spatial aggregates capture stroke/ink structure that individual pixels miss, boosting linear separability. Inference adds only ~50 integer adds per item. If dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 2× label-linear, the hypothesis fails.
```
```python
"""label-spatial: Linear model on spatially-augmented features for 8x8 digit images.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Features: 64 raw pixels + 8 row sums + 8 col sums + 16 2x2 block sums = 96 dims.
Exported weights: magic 0x53504154, d=64, D=96, C=10; float32 mean[D], std[D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_RAW = 64
D_AUG = 96  # 64 + 8 + 8 + 16


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def augment(X_raw):
    """X_raw: (n, 64) uint8. Returns (n, 96) float32 augmented features."""
    n = X_raw.shape[0]
    X = X_raw.astype(np.float32)
    # Row sums (8)
    row_sums = X.reshape(n, 8, 8).sum(axis=2)          # (n, 8)
    # Col sums (8)
    col_sums = X.reshape(n, 8, 8).sum(axis=1)          # (n, 8)
    # 2x2 block sums (16)
    block_sums = X.reshape(n, 4, 2, 4, 2).sum(axis=(2, 4)).reshape(n, 16)  # (n, 16)
    return np.concatenate([X, row_sums, col_sums, block_sums], axis=1).astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert X_raw.shape[1] == D_RAW

    X = augment(X_raw)
    n, D = X.shape
    assert D == D_AUG

    rng = np.random.default_rng(SEED)
    # Standardize per feature
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)

    # Linear model (multinomial logistic regression)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
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
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, d, D, C; mean[D], std[D], W[D*C], b[C]
    out = struct.pack("<4i", 0x53504154, D_RAW, D_AUG, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-spatial infer: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 pixels). Output: 1-byte class (0..9).
 * Weights format: magic 0x53504154, d=64, D=96, C=10; float32 mean[D], std[D], W[D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x53504154) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || D != 96 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (2L * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + D;
    const float *W    = std + D;
    const float *b    = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *sc   = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* 0..63: raw pixels as float */
        for (int j = 0; j < 64; j++) feat[j] = (float)xi[j];

        /* 64..71: row sums (8 rows) */
        for (int r = 0; r < 8; r++) {
            float sum = 0.0f;
            for (int c = 0; c < 8; c++) sum += (float)xi[r * 8 + c];
            feat[64 + r] = sum;
        }
        /* 72..79: col sums (8 cols) */
        for (int c = 0; c < 8; c++) {
            float sum = 0.0f;
            for (int r = 0; r < 8; r++) sum += (float)xi[r * 8 + c];
            feat[72 + c] = sum;
        }
        /* 80..95: 2x2 block sums (16 blocks) */
        int bidx = 0;
        for (int br = 0; br < 4; br++) {
            for (int bc = 0; bc < 4; bc++) {
                float sum = 0.0f;
                for (int dr = 0; dr < 2; dr++) {
                    for (int dc = 0; dc < 2; dc++) {
                        int r = br * 2 + dr;
                        int c = bc * 2 + dc;
                        sum += (float)xi[r * 8 + c];
                    }
                }
                feat[80 + bidx++] = sum;
            }
        }

        /* Standardize and compute scores */
        for (int j = 0; j < D; j++) feat[j] = (feat[j] - mean[j]) / std[j];
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int j = 0; j < D; j++) {
            float fj = feat[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) sc[c] += fj * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```