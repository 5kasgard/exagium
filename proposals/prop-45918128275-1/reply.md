```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ stores a small set of learned prototypes per class and classifies by nearest prototype. On 8x8 optdigits, prototypes capture stroke variations better than a linear boundary, while inference stays fast (30 prototypes × 64 dims ≈ 2k FLOPs). If dev accuracy ≤ 0.965 or Ir/item > 8000, the model does not beat label-linear meaningfully.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal features by z-scoring. Learns K prototypes per class via LVQ1.
Exports: magic 0x4C565131, d, K, C; float32 mean[d], std[d]; float32 proto[K*C][d]; uint8 proto_class[K*C]."""
import json, math, struct, sys

import numpy as np

K_PER_CLASS = 3
SEED = 1
LR0 = 0.1
EPOCHS = 80


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Z-score each feature (all ordinal per card)
    mean = X_bytes.mean(axis=0).astype(np.float32)
    std = X_bytes.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X = ((X_bytes.astype(np.float32) - mean) / std).astype(np.float32)

    # Initialize prototypes: K per class, sampled from that class
    protos = []
    proto_class = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(K_PER_CLASS, len(idx)), replace=False)
        protos.append(X[chosen])
        proto_class.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)
    proto_class = np.concatenate(proto_class).astype(np.uint8)
    n_protos = protos.shape[0]

    n = X.shape[0]
    for epoch in range(EPOCHS):
        lr = LR0 * (1.0 - epoch / EPOCHS)
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            yi = y[i]
            # squared Euclidean distance to all prototypes
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist2))
            if proto_class[best] == yi:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # Pack weights
    # magic, d, K, C
    hdr = struct.pack("<4i", 0x4C565131, d, K_PER_CLASS, C)
    body = mean.tobytes() + std.tobytes() + protos.tobytes() + proto_class.tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (nearest prototype). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C565131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    int n_protos = K * C;
    long expected = 16 + 2L * d * 4 + (long)n_protos * d * 4 + n_protos;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std = mean + d;
    const float *protos = (const float *)(std + d);
    const unsigned char *proto_class = (const unsigned char *)(protos + (long)n_protos * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * n_protos);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int p = 0; p < n_protos; p++) {
            float sum = 0.0f;
            const float *pr = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xnorm[j] - pr[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < n_protos; p++) if (dists[p] < dists[best]) best = p;
        unsigned char pred = proto_class[best];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```