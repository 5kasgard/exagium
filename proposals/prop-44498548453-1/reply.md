```hypothesis
Prototype-based LVQ classifier with 5 prototypes per class (50 total). Learns representative vectors per digit via deterministic LVQ; inference is nearest-prototype (squared Euclidean). Captures non-linear stroke shapes better than linear, with ~50*64=3200 FLOPs/item (comparable to label-linear). Fails if 0.9613 not beaten: optdigits needs more prototypes or richer distance.
```

```python
"""Prototype-based LVQ classifier for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports: magic 0x50524F54, d, P, C; then P records of (uint8 class, float32[d] proto)."""
import json, struct, sys

import numpy as np

SEED = 1
PROTOS_PER_CLASS = 5
EPOCHS = 20
LR_START = 0.3
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"]).astype(np.float32)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    P = PROTOS_PER_CLASS * C

    rng = np.random.default_rng(SEED)

    # Initialize prototypes: for each class, pick PROTOS_PER_CLASS random samples (or means with jitter)
    protos = np.zeros((P, d), dtype=np.float32)
    proto_class = np.zeros(P, dtype=np.uint8)
    p_idx = 0
    for c in range(C):
        class_mask = (y == c)
        class_samples = X[class_mask]
        if len(class_samples) == 0:
            # fallback: random noise around 8
            protos[p_idx:p_idx+PROTOS_PER_CLASS] = rng.normal(8, 2, (PROTOS_PER_CLASS, d)).astype(np.float32)
        else:
            # k-means++ style: first is mean, rest are farthest from existing
            mean = class_samples.mean(axis=0)
            protos[p_idx] = mean
            proto_class[p_idx] = c
            p_idx += 1
            for k in range(1, PROTOS_PER_CLASS):
                if len(class_samples) <= k:
                    # duplicate with jitter
                    protos[p_idx] = mean + rng.normal(0, 0.5, d).astype(np.float32)
                else:
                    # pick sample farthest from current prototypes of this class
                    diffs = class_samples[:, None, :] - protos[p_idx-PROTOS_PER_CLASS:p_idx][None, :, :]
                    dists = np.sum(diffs * diffs, axis=2)
                    min_dists = np.min(dists, axis=1)
                    farthest = np.argmax(min_dists)
                    protos[p_idx] = class_samples[farthest]
                proto_class[p_idx] = c
                p_idx += 1

    # LVQ training
    n = X.shape[0]
    for epoch in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / max(1, EPOCHS - 1))
        order = rng.permutation(n)
        for idx in order:
            x = X[idx]
            target = y[idx]
            # distances to all prototypes
            diffs = protos - x
            dists = np.sum(diffs * diffs, axis=1)
            nearest = np.argmin(dists)
            if proto_class[nearest] == target:
                protos[nearest] += lr * (x - protos[nearest])
            else:
                protos[nearest] -= lr * (x - protos[nearest])

    # Write weights
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<3i", magic, d, P)
    out += proto_class.tobytes()
    out += protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Prototype-based LVQ (nearest prototype) for label tasks.
 * ./infer weights.bin in.rec N
 * Weights: magic(0x50524F54), int32 d, int32 P, uint8 class[P], float32 proto[P][d].
 * Input record: d bytes. Output: 1 byte class of nearest prototype (squared L2). */
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
    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x50524F54) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int d = hdr[1];
    int P = hdr[2];
    if (d <= 0 || P <= 0 || d > 1024 || P > 10000) { fprintf(stderr, "bad weights: dims\n"); return 3; }
    size_t expected = 12 + (size_t)P + (size_t)P * d * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights: size %ld != %zu\n", nw, expected); return 3; }
    const uint8_t *proto_class = (const uint8_t *)(wb + 12);
    const float *protos = (const float *)(wb + 12 + P);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *buf = (float *)malloc((size_t)d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // convert to float
        for (int j = 0; j < d; j++) buf[j] = (float)xi[j];
        float best_dist = 1e30f;
        uint8_t best_class = 0;
        for (int p = 0; p < P; p++) {
            const float *pr = protos + (size_t)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = buf[j] - pr[j];
                dist += diff * diff;
                if (dist >= best_dist) break; // early exit
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_class[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```