```hypothesis
A prototype-based classifier (k-means per class, 4 prototypes/class = 40 total) learns compact class representatives.
It captures non-linear boundaries better than linear models while keeping inference cost low (40×64 distance ops/item).
If dev accuracy ≤ 0.9613 (label-linear), the prototype approach fails to justify its complexity for this task.
```
```python
import json, struct, sys

import numpy as np

PROTOS_PER_CLASS = 4
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(X_class, k, rng, max_iter=100):
    n, d = X_class.shape
    if n <= k:
        protos = np.zeros((k, d), dtype=np.float32)
        protos[:n] = X_class
        return protos
    idx = rng.choice(n, k, replace=False)
    protos = X_class[idx].copy()
    for _ in range(max_iter):
        diff = X_class[:, None, :] - protos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_protos = np.zeros_like(protos)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(n):
            a = assign[i]
            new_protos[a] += X_class[i]
            counts[a] += 1
        for j in range(k):
            if counts[j] > 0:
                new_protos[j] /= counts[j]
            else:
                new_protos[j] = X_class[rng.integers(n)]
        if np.allclose(protos, new_protos):
            break
        protos = new_protos
    return protos


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    rng = np.random.default_rng(SEED)
    X = X_bytes.astype(np.float32)

    all_protos = []
    proto_classes = []
    for c in range(C):
        mask = (y == c)
        if not np.any(mask):
            continue
        Xc = X[mask]
        protos = kmeans_class(Xc, PROTOS_PER_CLASS, rng)
        all_protos.append(protos)
        proto_classes.extend([c] * PROTOS_PER_CLASS)

    protos = np.vstack(all_protos)
    proto_classes = np.array(proto_classes, dtype=np.uint8)
    P = protos.shape[0]

    out = struct.pack("<4i", 0x50524F54, d, P, C)
    out += protos.astype("<f4").tobytes()
    out += proto_classes.tobytes()
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d != 64 || C != 10 || P <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)P * d * 4 + (long)P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *protos = (const float *)(hd + 4);
    const unsigned char *proto_classes = (const unsigned char *)(protos + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_float = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) x_float[j] = (float)xi[j];

        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < P; p++) {
            const float *proto = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = x_float[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_classes[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```