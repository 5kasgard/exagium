```hypothesis
Family: label-lvq (Learning Vector Quantization). Optdigits has locally smooth class boundaries in pixel space; LVQ places a few prototypes per class (e.g., 3) and classifies by nearest prototype (L2). This captures non-linear boundaries better than linear while keeping inference at ~30×64 FLOPs/item. If dev accuracy ≤ 0.961 (linear) or Ir/item ≥ 8000, the hypothesis fails.
```

```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Writes weights.bin with magic 0x4C565131.
Format: uint32 magic, n_prototypes_per_class K, C, d; float32 prototypes[C*K*d]."""
import json, math, struct, sys

import numpy as np

K = 3          # prototypes per class
SEED = 1
EPOCHS = 30
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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Initialize prototypes as class means + small noise
    proto = np.zeros((C, K, d), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            proto[c] = rng.normal(0, 1, (K, d)).astype(np.float32)
        else:
            mean = Xc.mean(axis=0)
            for k in range(K):
                proto[c, k] = mean + rng.normal(0, 0.1, d).astype(np.float32)

    # LVQ training
    for epoch in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / max(1, EPOCHS - 1))
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            target = y[i]
            # squared distances to all prototypes
            diff = proto.reshape(C * K, d) - x
            dist2 = np.sum(diff * diff, axis=1)
            winner = int(np.argmin(dist2))
            w_class = winner // K
            w_proto = winner % K
            if w_class == target:
                proto[w_class, w_proto] += lr * (x - proto[w_class, w_proto])
            else:
                proto[w_class, w_proto] -= lr * (x - proto[w_class, w_proto])

    # Flatten prototypes to (C*K*d)
    proto_flat = proto.reshape(-1).astype(np.float32)
    out = struct.pack("<4I", 0x4C565131, K, C, d) + proto_flat.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Weights: uint32 magic(0x4C565131), K, C, d; float32 proto[C*K*d]
 * Input record: d bytes. Output: 1 byte class (nearest prototype, first max wins). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t K = hd[1], C = hd[2], d = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 4096) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + (size_t)C * K * d * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *proto = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xbuf = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        for (uint32_t j = 0; j < d; j++) xbuf[j] = (float)xi[j];
        float best_dist = 1e30f; unsigned char best_class = 0;
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t kk = 0; kk < K; kk++) {
                const float *p = proto + ((size_t)c * K + kk) * d;
                float dist = 0.0f;
                for (uint32_t j = 0; j < d; j++) {
                    float diff = xbuf[j] - p[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```