```hypothesis
Family: label-lvq (Learning Vector Quantization). One prototype per class initialized at class means in z-scored space, refined by LVQ (pull toward same-class samples, push from different-class). Optdigits has consistent stroke patterns per digit; prototypes capture typical shapes with non-linear Voronoi boundaries. Fast inference: 10×64 multiply-adds per item. If dev accuracy ≤ label-linear (0.9613), prototypes are too simple for intra-class variation.
```

```python
"""LVQ family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Z-scores ordinal features, learns one prototype per class via LVQ.
Exports: magic 0x4C565131, d, P, C; float32 mean[d], std[d]; float32 proto[P][d]; uint8 proto_class[P]."""
import json, struct, sys
import numpy as np

SEED, LR, EPOCHS = 1, 0.1, 10


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

    # Z-score ordinal features (all are ordinal per card)
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # One prototype per class initialized at class mean
    P = C
    proto = np.zeros((P, d), np.float32)
    proto_class = np.zeros(P, np.uint8)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            proto[c] = Z[mask].mean(axis=0)
        else:
            proto[c] = rng.normal(size=d).astype(np.float32)
        proto_class[c] = c

    # LVQ epochs
    n = Z.shape[0]
    order = np.arange(n)
    for ep in range(EPOCHS):
        rng.shuffle(order)
        lr = LR * (1.0 - ep / EPOCHS)
        for i in order:
            z = Z[i]
            dists = np.sum((proto - z) ** 2, axis=1)
            best = int(np.argmin(dists))
            if proto_class[best] == y[i]:
                proto[best] += lr * (z - proto[best])
            else:
                proto[best] -= lr * (z - proto[best])

    # Pack weights
    out = struct.pack("<4i", 0x4C565131, d, P, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += proto.astype("<f4").tobytes()
    out += proto_class.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* LVQ family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class of nearest prototype. */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long need = 16 + 4L * d * 2 + 4L * P * d + P;
    if (nw != need) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *proto = std + d;
    const unsigned char *proto_class = (const unsigned char *)(proto + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *znorm = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) znorm[j] = (xi[j] - mean[j]) / std[j];
        float best_dist = 1e30f; int best_idx = 0;
        for (int p = 0; p < P; p++) {
            float dist = 0.0f;
            const float *pr = proto + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = znorm[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_idx = p; }
        }
        unsigned char cls = proto_class[best_idx];
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```