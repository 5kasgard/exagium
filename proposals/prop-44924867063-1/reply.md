```hypothesis
Family: label-lvq (Learning Vector Quantization with 5 prototypes/class). Optdigits has non-linear class boundaries in the 64-dim pixel space; LVQ learns compact, class-conditional prototypes that adapt to local structure, unlike the global linear separator in label-linear. With 50 prototypes × 64 dims, inference stays cheap (~3.2k FLOPs/item). Failure: dev accuracy ≤ 96.13% (label-linear's score).
```
```python
import json, struct, sys

import numpy as np

PROTOS_PER_CLASS = 5
EPOCHS = 20
INIT_LR = 0.1
SEED = 1


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

    # Z-score per pixel (all ordinal)
    means = X.mean(axis=0).astype(np.float32)
    stds = X.std(axis=0).astype(np.float32)
    stds[stds == 0] = 1.0
    Xz = ((X - means) / stds).astype(np.float32)

    P = PROTOS_PER_CLASS * C
    protos = np.zeros((P, d), dtype=np.float32)
    proto_labels = np.zeros(P, dtype=np.uint8)

    # Deterministic prototype init per class via 1D PCA sorting
    for c in range(C):
        Xc = Xz[y == c]
        if len(Xc) == 0:
            continue
        if len(Xc) <= PROTOS_PER_CLASS:
            protos[c * PROTOS_PER_CLASS:(c + 1) * PROTOS_PER_CLASS, :len(Xc)] = Xc[:PROTOS_PER_CLASS]
        else:
            Xc_centered = Xc - Xc.mean(axis=0)
            U, S, Vt = np.linalg.svd(Xc_centered, full_matrices=False)
            pc1 = Vt[0]
            proj = Xc_centered @ pc1
            order = np.argsort(proj)
            Xc_sorted = Xc[order]
            groups = np.array_split(Xc_sorted, PROTOS_PER_CLASS)
            for i, g in enumerate(groups):
                protos[c * PROTOS_PER_CLASS + i] = g.mean(axis=0)
        proto_labels[c * PROTOS_PER_CLASS:(c + 1) * PROTOS_PER_CLASS] = c

    # LVQ training
    n = Xz.shape[0]
    lr = INIT_LR
    for epoch in range(EPOCHS):
        epoch_rng = np.random.default_rng(SEED + epoch + 1000)
        order = epoch_rng.permutation(n)
        for idx in order:
            x = Xz[idx]
            target = y[idx]
            dists = np.sum((protos - x) ** 2, axis=1)
            nearest = int(np.argmin(dists))
            if proto_labels[nearest] == target:
                protos[nearest] += lr * (x - protos[nearest])
            else:
                protos[nearest] -= lr * (x - protos[nearest])
        lr *= 0.95

    # Save: magic, d, P, C, means[d], stds[d], protos[P*d], proto_labels[P]
    out = struct.pack("<4i", 0x4C565131, d, P, C)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (C < 1 || C > 256 || P < 1) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * P * d + 1L * P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *means = (const float *)(hd + 4);
    const float *stds = means + d;
    const float *protos = (const float *)(stds + d);
    const unsigned char *proto_labels = (const unsigned char *)(protos + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - means[j]) / stds[j];

        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < P; p++) {
            float dist = 0.0f;
            const float *proto = protos + (long)p * d;
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