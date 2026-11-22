```hypothesis
Family: label-lvq (Learning Vector Quantization with one prototype per class).
Why: Optdigits 8x8 images have distinct class clusters; LVQ adapts prototypes to boundaries, often beating linear models on small image datasets. Inference is just 10 Euclidean distances (640 multiply-adds), far cheaper than label-linear's table lookups.
Failure: If prototypes collapse or classes overlap heavily, accuracy drops below 0.9613 (label-linear dev).
```
```python
import json, struct, sys

import numpy as np

SEED = 1
EPOCHS = 30
INIT_ALPHA = 0.1
ALPHA_DECAY = 0.99


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

    # Normalize per feature
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xn = (X.astype(np.float32) - mu) / sd

    # Initialize prototypes as class means
    protos = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if np.any(mask):
            protos[c] = Xn[mask].mean(axis=0)
        else:
            protos[c] = 0.0

    # LVQ training
    n = Xn.shape[0]
    alpha = INIT_ALPHA
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for idx in order:
            x = Xn[idx]
            target = y[idx]
            dists = np.sum((protos - x) ** 2, axis=1)
            winner = np.argmin(dists)
            if winner == target:
                protos[winner] += alpha * (x - protos[winner])
            else:
                protos[winner] -= alpha * (x - protos[winner])
        alpha *= ALPHA_DECAY

    # Write weights.bin: magic, d, C; then mu[d], sd[d], protos[C][d] as float32 LE
    out = struct.pack("<3i", 0x4C565131, d, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 12 + (long)d * 4 * 2 + (long)C * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mu = (const float *)(hd + 3);
    const float *sd = mu + d;
    const float *protos = sd + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float dist = 0.0f;
            const float *pc = protos + (long)c * d;
            for (int j = 0; j < d; j++) {
                float xnorm = (xi[j] - mu[j]) / sd[j];
                float diff = xnorm - pc[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```