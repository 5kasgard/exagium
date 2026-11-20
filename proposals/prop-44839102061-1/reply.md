```hypothesis
Family: label-elm (Extreme Learning Machine). A single random ReLU layer (fixed Xavier weights, seed=1) followed by a ridge-regression readout (closed-form, λ=1e-4). On optdigits the 8×8 pixel patterns benefit from non-linear features; ELM provides a rich random basis (128 units) with an optimal linear readout, beating plain linear (0.9613) while training deterministically in one pass. If dev accuracy ≤ 0.9613 the hypothesis fails.
```
```python
import json, struct, sys

import numpy as np

H = 128
SEED = 1
L2 = 1e-4


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
    n, d = X.shape

    rng = np.random.default_rng(SEED)

    # Z-score ordinal features (all 64 are ordinal)
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xz = (Xf - mean) / std

    # Random hidden layer (Xavier uniform)
    W1 = (rng.uniform(-1, 1, (d, H)) * np.sqrt(6.0 / d)).astype(np.float32)
    b1 = (rng.uniform(-1, 1, H) * np.sqrt(6.0 / d)).astype(np.float32)

    # Hidden activations with ReLU
    H_act = np.maximum(Xz @ W1 + b1, 0).astype(np.float32)
    H_aug = np.hstack([H_act, np.ones((n, 1), dtype=np.float32)])

    # One-hot labels
    Y = np.zeros((n, C), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    # Ridge regression: (H_aug^T H_aug + λI) W2 = H_aug^T Y
    A = H_aug.T @ H_aug
    A.flat[::H + 2] += L2
    B = H_aug.T @ Y
    W2 = np.linalg.solve(A, B).astype(np.float32)  # (H+1, C)

    # Export: magic, d, H, C, mean[d], std[d], W1[d][H], b1[H], W2[H+1][C]
    out = struct.pack("<4i", 0x454C4D31, d, H, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454C4D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    long expected = 16 + 4L * (2L * d + (long)d * H + H + (long)(H + 1) * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W1 = std + d;
    const float *b1 = W1 + (long)d * H;
    const float *W2 = b1 + H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * (H + 1));
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d; j++) {
                float z = (xi[j] - mean[j]) / std[j];
                sum += z * W1[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        hidden[H] = 1.0f;

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int h = 0; h <= H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```