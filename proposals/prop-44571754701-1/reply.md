```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of discriminative prototypes per class (e.g., 5 per class = 50 total), capturing multi-modal digit shapes better than a single linear boundary. Inference is a simple nearest-prototype search in normalized pixel space — fast, no transcendental ops, and deterministic. Expected to beat label-linear (0.9613) and approach label-mlp16 (0.9770) at lower CPU cost. Fails if dev accuracy ≤ 0.9613 or instructions/item exceed label-mlp16.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, stdlib+numpy only.
Encodes ordinal features via per-byte z-scoring (like label-linear).
Learns 5 prototypes per class via LVQ1 with decaying learning rate.
Exports: magic, d, P, C; float32 mean[d], std[d]; float32 proto[P][d]; uint8 proto_label[P]."""
import json, math, struct, sys

import numpy as np

SEED = 1
PROTOS_PER_CLASS = 5
EPOCHS = 30
INIT_LR = 0.1

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
    n = X.shape[0]

    # Per-feature z-scoring (ordinal only per card)
    means = np.zeros(d, np.float64)
    stds = np.ones(d, np.float64)
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            means[j] = v.mean()
            stds[j] = v.std() or 1.0
    Xn = (X.astype(np.float64) - means) / stds
    Xn = Xn.astype(np.float32)

    rng = np.random.default_rng(SEED)

    # Initialize prototypes: pick PROTOS_PER_CLASS random samples per class
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(PROTOS_PER_CLASS, len(idx)), replace=False)
        protos.append(Xn[chosen])
        proto_labels.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)  # [P, d]
    proto_labels = np.concatenate(proto_labels).astype(np.uint8)  # [P]
    P = protos.shape[0]

    # LVQ1 training
    for epoch in range(EPOCHS):
        lr = INIT_LR * (1.0 - epoch / EPOCHS)
        order = rng.permutation(n)
        for i in order:
            x = Xn[i]
            yi = y[i]
            # squared Euclidean distances
            diffs = protos - x  # [P, d]
            dists = np.sum(diffs * diffs, axis=1)
            best = np.argmin(dists)
            if proto_labels[best] == yi:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # Pack weights: magic, d, P, C; mean[d], std[d]; proto[P][d]; proto_label[P]
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, P, C)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_labels.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: 1 byte class (nearest prototype).
 * Weights format: magic(0x4C565131), d, P, C (int32); mean[d], std[d] (float32); proto[P][d] (float32); proto_label[P] (uint8). */
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
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * (long)P * d + P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *protos = std + d;
    const unsigned char *proto_labels = (const unsigned char *)(protos + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xn = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xn[j] = (xi[j] - mean[j]) / std[j];
        float best_dist = 1e30f; unsigned char best_label = 0;
        for (int p = 0; p < P; p++) {
            const float *pr = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = xn[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_label = proto_labels[p]; }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```