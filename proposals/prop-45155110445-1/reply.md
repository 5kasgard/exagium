```hypothesis
Family: label-proto (nearest centroid / prototypical classifier with z-score normalization).
Why: Optdigits pixels are ordinal 0..16; class-conditional means are strong discriminants for digits.
A nearest-centroid classifier on z-scored features is deterministic, tiny (weights ~3 KB), and inference
cost is ~10×64 FLOPs/item (lower than label-linear's table lookups). It should match or beat
label-linear's 96.1% dev accuracy; if it falls below ~95.5% the hypothesis fails.
```
```python
"""label-proto: nearest-centroid classifier with per-feature z-scoring.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Weights format (little-endian):
  uint32 magic=0x50524F54 ('PROT'), int32 d, int32 C,
  float32 feature_mean[d], float32 feature_std[d], float32 proto[C][d]"""
import json, struct, sys
import numpy as np

MAGIC = 0x50524F54  # 'PROT'

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
    d = card["input"]["bytes"]
    C = card["classes"]

    # Per-feature z-score on training set
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # Class prototypes (mean of normalized features)
    proto = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            proto[c] = Xz[mask].mean(axis=0).astype(np.float32)
        else:
            proto[c] = 0.0

    # Pack weights
    out = struct.pack("<Iii", MAGIC, d, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += proto.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-proto infer: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte class index (argmin squared Euclidean distance). */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x50524F54) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t C = *(int32_t*)(wb + 8);
    if (d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 12 + (size_t)d * 4 * 2 + (size_t)C * d * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *fmu = (const float*)(wb + 12);
    const float *fsd = fmu + d;
    const float *proto = fsd + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);
    float *dist = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - fmu[j]) / fsd[j];

        // squared Euclidean distance to each prototype
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *pc = proto + (size_t)c * d;
            for (int j = 0; j < d; j++) {
                float diff = xz[j] - pc[j];
                sum += diff * diff;
            }
            dist[c] = sum;
        }

        // argmin (first min wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (dist[c] < dist[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```