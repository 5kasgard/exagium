```hypothesis
Family: label-adjpoly. Adds pairwise products of spatially adjacent pixels (4-neighborhood on the 8x8 grid) to a multinomial logistic regression, giving 176 features (64 raw + 112 interactions). This captures local stroke correlations that a purely linear model misses, while staying extremely cheap at inference (≈1760 multiply-adds/item). If dev accuracy does not exceed label-linear (0.9613), it shows that second-order spatial features are insufficient for this task.
```

```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x41444A50  # "ADJP"

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
    assert d == 64, "Expect 8x8 input"

    # adjacency pairs for 8x8 grid (right and down)
    pairs = []
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            if j < 7:
                pairs.append((idx, idx + 1))
            if i < 7:
                pairs.append((idx, idx + 8))
    num_pairs = len(pairs)  # 112
    num_feat = d + num_pairs  # 176

    rng = np.random.default_rng(SEED)

    # Build feature matrix
    F = np.empty((X.shape[0], num_feat), dtype=np.float32)
    F[:, :d] = X.astype(np.float32)
    for p, (j1, j2) in enumerate(pairs):
        F[:, d + p] = X[:, j1].astype(np.float32) * X[:, j2].astype(np.float32)

    # Per-feature standardization
    mean = F.mean(axis=0).astype(np.float32)
    std = F.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    F = (F - mean) / std

    n, k = F.shape
    # Xavier-like init for linear weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            f, yy = F[idx], y[idx]
            # softmax gradient
            S = f @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)
            grads = [f.T @ G + L2 * W, G.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, d, num_feat, C; mean[num_feat]; std[num_feat]; W[num_feat*C]; b[C]
    out = struct.pack("<Iiii", MAGIC, d, num_feat, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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

/* 8x8 adjacency pairs (right, down) -- 112 pairs */
static const uint8_t adj_j1[112] = {
    0,1,2,3,4,5,6,  8,9,10,11,12,13,14,  16,17,18,19,20,21,22,
    24,25,26,27,28,29,30,  32,33,34,35,36,37,38,  40,41,42,43,44,45,46,
    48,49,50,51,52,53,54,  56,57,58,59,60,61,62,  /* right neighbors */
    0,1,2,3,4,5,6,7,  8,9,10,11,12,13,14,15,  16,17,18,19,20,21,22,23,
    24,25,26,27,28,29,30,31,  32,33,34,35,36,37,38,39,  40,41,42,43,44,45,46,47,
    48,49,50,51,52,53,54,55,  56,57,58,59,60,61,62,63  /* down neighbors */
};
static const uint8_t adj_j2[112] = {
    1,2,3,4,5,6,7,  9,10,11,12,13,14,15,  17,18,19,20,21,22,23,
    25,26,27,28,29,30,31,  33,34,35,36,37,38,39,  41,42,43,44,45,46,47,
    49,50,51,52,53,54,55,  57,58,59,60,61,62,63,  /* right */
    8,9,10,11,12,13,14,15,  16,17,18,19,20,21,22,23,  24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,  40,41,42,43,44,45,46,47,  48,49,50,51,52,53,54,55,
    56,57,58,59,60,61,62,63  /* down */
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x41444A50) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int32_t)hdr[1], num_feat = (int32_t)hdr[2], C = (int32_t)hdr[3];
    if (d != 64 || num_feat != 176 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (num_feat * 2 + num_feat * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hdr + 4);
    const float *std  = mean + num_feat;
    const float *W    = std + num_feat;
    const float *b    = W + (long)num_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * num_feat);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* raw pixels */
        for (int j = 0; j < d; j++) feat[j] = (xi[j] - mean[j]) / std[j];
        /* adjacent products */
        for (int p = 0; p < 112; p++) {
            float prod = (float)xi[adj_j1[p]] * (float)xi[adj_j2[p]];
            feat[d + p] = (prod - mean[d + p]) / std[d + p];
        }

        /* scores = b + feat * W */
        memcpy(scores, b, C * sizeof(float));
        for (int f = 0; f < num_feat; f++) {
            const float *wrow = W + (long)f * C;
            float fv = feat[f];
            for (int c = 0; c < C; c++) scores[c] += fv * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```