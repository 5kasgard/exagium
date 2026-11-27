```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of prototypes per class, capturing non-linear boundaries in the 8×8 pixel space. It should exceed label-linear's 0.961 accuracy while keeping inference cost low (~30 prototypes × 64 dims = 1920 ops vs linear's 640 adds). Failure if dev accuracy ≤ 0.961 or Ir/item > 6000.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. LVQ1 with 3 prototypes/class, 30 epochs, alpha 0.1->0.01.
Weights: magic 0x4C565131, d, M, C; then M*C*d float32 prototypes (row-major per class)."""
import json, math, struct, sys

import numpy as np

SEED = 1
M = 3               # prototypes per class
EPOCHS = 30
ALPHA_START = 0.1
ALPHA_END = 0.01


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

    # Initialize prototypes: pick M random samples from each class (deterministic)
    protos = np.zeros((C * M, d), dtype=np.float32)
    proto_class = np.zeros(C * M, dtype=np.int64)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(M, len(idx)), replace=False)
        for m, xi in enumerate(chosen):
            protos[c * M + m] = X[xi].astype(np.float32)
            proto_class[c * M + m] = c

    Xf = X.astype(np.float32)

    # LVQ1 training
    for epoch in range(EPOCHS):
        alpha = ALPHA_START * (ALPHA_END / ALPHA_START) ** (epoch / max(1, EPOCHS - 1))
        order = rng.permutation(n)
        for i in order:
            x = Xf[i]
            yi = y[i]
            # squared Euclidean distances
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist2))
            if proto_class[best] == yi:
                protos[best] += alpha * (x - protos[best])
            else:
                protos[best] -= alpha * (x - protos[best])

    # Write weights
    # magic, d, M, C
    out = struct.pack("<4i", 0x4C565131, d, M, C)
    out += protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output: 1 byte class (argmin squared Euclidean distance to prototypes). */
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
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || M < 1 || d < 1 ||
        nw != 16 + (long)C * M * d * 4) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *protos = (const float *)(hd + 4);
    int P = C * M;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < P; p++) {
            const float *w = protos + (long)p * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float delta = xi[j] - w[j];
                dist2 += delta * delta;
            }
            if (dist2 < best_dist) {
                best_dist = dist2;
                best_class = (unsigned char)(p / M);
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```