```hypothesis
Family: label-poly2local — logistic regression with 112 local pairwise interactions (adjacent pixels in 8×8 grid) added to 64 z-scored pixels.
Why: Digits are strokes; adjacent-pixel products capture oriented edges/curves that linear pixels miss, boosting accuracy with only 176 features.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ 8000 Ir/item (label-mlp16), the family does not improve the Pareto frontier.
```
```python
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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Z-score each pixel
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = (X.astype(np.float64) - mu) / sd

    # Adjacent pairs in 8x8 grid: 56 horizontal + 56 vertical = 112
    pairs = []
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if c < 7:
                pairs.append((idx, idx + 1))
            if r < 7:
                pairs.append((idx, idx + 8))

    Z_inter = Z[:, [p[0] for p in pairs]] * Z[:, [p[1] for p in pairs]]
    Z_all = np.concatenate([Z, Z_inter], axis=1).astype(np.float32)
    n, k = Z_all.shape

    # Logistic regression with Adam
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_all[idx], y[idx]

            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
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

    # Export: magic, d=64, k=176, C=10, mu[64], inv_sd[64], W[176*10], b[10]
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, k, C)
    out += mu.astype("<f4").tobytes()
    out += (1.0 / sd).astype("<f4").tobytes()
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

static const uint8_t pairs[112][2] = {
    {0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},
    {8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},
    {16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},
    {24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},
    {32,33},{33,34},{34,35},{35,36},{36,37},{37,38},{38,39},
    {40,41},{41,42},{42,43},{43,44},{44,45},{45,46},{46,47},
    {48,49},{49,50},{50,51},{51,52},{52,53},{53,54},{54,55},
    {56,57},{57,58},{58,59},{59,60},{60,61},{61,62},{62,63},
    {0,8},{8,16},{16,24},{24,32},{32,40},{40,48},{48,56},
    {1,9},{9,17},{17,25},{25,33},{33,41},{41,49},{49,57},
    {2,10},{10,18},{18,26},{26,34},{34,42},{42,50},{50,58},
    {3,11},{11,19},{19,27},{27,35},{35,43},{43,51},{51,59},
    {4,12},{12,20},{20,28},{28,36},{36,44},{44,52},{52,60},
    {5,13},{13,21},{21,29},{29,37},{37,45},{45,53},{53,61},
    {6,14},{14,22},{22,30},{30,38},{38,46},{46,54},{54,62},
    {7,15},{15,23},{23,31},{31,39},{39,47},{47,55},{55,63}
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw != 7608) { fprintf(stderr, "bad weights size %ld != 7608\n", nw); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x504F4C32 || hd[1] != 64 || hd[2] != 176 || hd[3] != 10) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    const float *mu = (const float *)(hd + 4);
    const float *inv_sd = mu + 64;
    const float *W = inv_sd + 64;
    const float *b = W + 176 * 10;

    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float z[64];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        for (int j = 0; j < 64; j++) {
            z[j] = (xi[j] - mu[j]) * inv_sd[j];
        }

        for (int c = 0; c < 10; c++) scores[c] = b[c];
        for (int j = 0; j < 64; j++) {
            float zv = z[j];
            for (int c = 0; c < 10; c++) scores[c] += zv * W[j * 10 + c];
        }

        for (int p = 0; p < 112; p++) {
            float zv = z[pairs[p][0]] * z[pairs[p][1]];
            int w_idx = (64 + p) * 10;
            for (int c = 0; c < 10; c++) scores[c] += zv * W[w_idx + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```