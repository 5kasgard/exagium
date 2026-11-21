```hypothesis
Family: label-lvq (Learning Vector Quantization). Uses 10 prototypes per class (100 total) trained by competitive learning on standardized pixels. Optdigits' low-dimensional ordinal features suit prototype-based classification; LVQ learns non-linear boundaries with few parameters. If dev accuracy ≤ 0.9613 (label-linear), the prototype approach lacks capacity for this task.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
N_PROTOS_PER_CLASS = 10
EPOCHS = 30
LR_START = 0.1
LR_END = 0.01

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
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Standardize
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xz = (Xf - mean) / std

    # Initialize prototypes: pick N_PROTOS_PER_CLASS random samples per class
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=N_PROTOS_PER_CLASS, replace=False)
        protos.append(Xz[chosen])
        proto_labels.append(np.full(N_PROTOS_PER_CLASS, c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)  # (P, d)
    proto_labels = np.concatenate(proto_labels)    # (P,)
    P = protos.shape[0]

    # LVQ training
    for epoch in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / (EPOCHS - 1)) if EPOCHS > 1 else LR_START
        order = rng.permutation(n)
        for idx in order:
            x = Xz[idx]
            target = y[idx]
            # Find nearest prototype
            diffs = protos - x  # (P, d)
            dists = np.sum(diffs * diffs, axis=1)
            nearest = int(np.argmin(dists))
            if proto_labels[nearest] == target:
                protos[nearest] += lr * (x - protos[nearest])
            else:
                protos[nearest] -= lr * (x - protos[nearest])

    # Save weights
    # Magic: 0x4C565131 ("LVQ1")
    # Format: magic, P, d, C; mean[d], std[d]; protos[P*d]; proto_labels[P]
    out = struct.pack("<4i", 0x4C565131, P, d, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_labels.astype("<u1").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int P = hd[1], d = hd[2], C = hd[3];
    if (P <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + (size_t)d * 4 * 2 + (size_t)P * d * 4 + (size_t)P;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std = mean + d;
    const float *protos = (const float *)(std + d);
    const unsigned char *proto_labels = (const unsigned char *)(protos + (size_t)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            xz[j] = (xi[j] - mean[j]) / std[j];
        }
        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < P; p++) {
            const float *proto = protos + (size_t)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = xz[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_label = proto_labels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```