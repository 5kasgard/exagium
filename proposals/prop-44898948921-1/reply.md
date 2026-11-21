```hypothesis
Family: label-lvq (Learning Vector Quantization with multiple prototypes per class).
Why: Optdigits has non-linear class boundaries in pixel space; LVQ learns prototype vectors per class that adapt to local structure, outperforming a single linear separator. With 4 prototypes/class (40 total), inference is just 40 distance computations (64-dim), far cheaper than k-NN (all train items) and competitive with linear cost. Deterministic LVQ (fixed seed, shuffled epochs) fits the sandbox.
Failure: Dev accuracy <= 0.9613 (label-linear) or Ir/item > 10000.
```

```python
"""label-lvq: Learning Vector Quantization for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin (deterministic: seed 1, single thread)
Encodes ordinal bytes via per-feature z-score (mean/std from train). Learns P prototypes per class via LVQ1.
Exports: magic 0x4C565131, d, n_proto, C; float32 mean[d], std[d]; float32 proto[n_proto][d]; uint8 proto_label[n_proto]."""
import json, math, struct, sys

import numpy as np

PROTOS_PER_CLASS = 4
SEED, EPOCHS, INIT_LR = 1, 50, 0.1


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
    rng = np.random.default_rng(SEED)

    # Per-feature z-score (ordinal only per card)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)  # (n, d)

    # Initialize prototypes: pick PROTOS_PER_CLASS random samples per class
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(PROTOS_PER_CLASS, len(idx)), replace=False)
        protos.append(Z[chosen])
        proto_labels.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)  # (n_proto, d)
    proto_labels = np.concatenate(proto_labels).astype(np.uint8)
    n_proto = protos.shape[0]

    # LVQ1 training
    lr = INIT_LR
    n = Z.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for i in order:
            z = Z[i]
            yi = y[i]
            # squared Euclidean distances
            diff = protos - z  # (n_proto, d)
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            if proto_labels[best] == yi:
                protos[best] += lr * (z - protos[best])
            else:
                protos[best] -= lr * (z - protos[best])
        lr *= 0.99  # decay

    # Pack weights
    # header: magic, d, n_proto, C (4 int32)
    hdr = struct.pack("<4i", 0x4C565131, d, n_proto, C)
    # mean[d], std[d] (float32)
    body = mean.astype("<f4").tobytes() + std.astype("<f4").tobytes()
    # protos[n_proto][d] (float32)
    body += protos.astype("<f4").tobytes()
    # proto_labels[n_proto] (uint8)
    body += proto_labels.tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```

```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Preprocess with stored mean/std, find nearest prototype (squared Euclidean),
 * output its class label (1 byte). Weights format: see train.py. */
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
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d <= 0 || n_proto <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * n_proto * d + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *protos = std + d;
    const unsigned char *proto_labels = (const unsigned char *)(protos + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *zbuf = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // preprocess: z = (x - mean) / std
        for (int j = 0; j < d; j++) zbuf[j] = (xi[j] - mean[j]) / std[j];
        // find nearest prototype
        float best_dist = 1e30f; int best_idx = 0;
        for (int p = 0; p < n_proto; p++) {
            const float *pr = protos + (long)p * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = zbuf[j] - pr[j];
                dist2 += diff * diff;
            }
            if (dist2 < best_dist) { best_dist = dist2; best_idx = p; }
        }
        unsigned char out = proto_labels[best_idx];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```