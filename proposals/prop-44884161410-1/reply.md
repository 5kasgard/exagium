```hypothesis
Family: label-lvq (Learning Vector Quantization). Optdigits has non-linear class boundaries in 64D pixel space; LVQ learns a small set of adaptive prototypes per class (3/class = 30 total), capturing local structure better than a global linear model while being far more parametric and faster than k-NN. With 30 prototypes, inference is ~30×64 multiply-adds, cheaper than label-mlp16's two matrix multiplies. Expected dev accuracy > 0.961 (label-linear) and approaching 0.977 (label-mlp16). Failure if dev accuracy ≤ 0.961 or instructions/item exceed label-mlp16.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, numpy only.
Learns 3 prototypes per class via LVQ1 with decaying learning rate.
Exports: magic 0x3151564C, n_proto, d, C; float32 protos[n_proto][d], uint8 labels[n_proto]."""
import json, struct, sys

import numpy as np

SEED = 1
PROTOS_PER_CLASS = 3
EPOCHS = 30
INIT_ALPHA = 0.1
FINAL_ALPHA = 0.01


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    Xf = X.astype(np.float32)

    # Initialize prototypes: pick PROTOS_PER_CLASS random samples from each class
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(PROTOS_PER_CLASS, len(idx)), replace=False)
        protos.append(Xf[chosen])
        proto_labels.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)          # (n_proto, d)
    proto_labels = np.concatenate(proto_labels)            # (n_proto,)
    n_proto = protos.shape[0]

    # LVQ1 training
    for epoch in range(EPOCHS):
        alpha = INIT_ALPHA * (FINAL_ALPHA / INIT_ALPHA) ** (epoch / (EPOCHS - 1)) if EPOCHS > 1 else INIT_ALPHA
        order = rng.permutation(n)
        for idx in order:
            x = Xf[idx]
            y_true = y[idx]
            # squared Euclidean distances
            diffs = protos - x
            dists = np.sum(diffs * diffs, axis=1)
            nearest = int(np.argmin(dists))
            if proto_labels[nearest] == y_true:
                protos[nearest] += alpha * (x - protos[nearest])
            else:
                protos[nearest] -= alpha * (x - protos[nearest])

    # Pack weights: magic, n_proto, d, C; then protos (float32), labels (uint8)
    magic = 0x3151564C  # "LVQ1" little-endian
    out = struct.pack("<4i", magic, n_proto, d, C)
    out += protos.astype("<f4").tobytes()
    out += proto_labels.astype("<B").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input: d-byte records (pixel counts). Output: 1-byte class index per record.
 * Weights format: magic(0x3151564C), n_proto, d, C (int32); float32 protos[n_proto][d]; uint8 labels[n_proto]. */
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
    if (nw < 16 || hd[0] != 0x3151564C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_proto = hd[1], d = hd[2], C = hd[3];
    if (n_proto <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * (long)n_proto * d + (long)n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *protos = (const float *)(hd + 4);
    const unsigned char *labels = (const unsigned char *)(protos + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xfloat = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * n_proto);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xfloat[j] = (float)xi[j];

        for (int p = 0; p < n_proto; p++) {
            const float *proto = protos + (long)p * d;
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = xfloat[j] - proto[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }

        int best = 0;
        for (int p = 1; p < n_proto; p++)
            if (dists[p] < dists[best]) best = p;
        unsigned char pred = labels[best];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```