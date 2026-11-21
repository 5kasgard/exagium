```hypothesis
Family: label-proto (Learning Vector Quantization prototype classifier).
Why: Optdigits has distinct per-class shapes; LVQ learns a small set of prototypical 8x8 patterns per class.
Non-linear decision boundaries beat linear (0.9613) with far fewer distance calculations than k-NN (30 prototypes vs 2867 train items).
Failure signal: dev accuracy ≤ 0.962 or Ir/item > 15000.
```
```python
"""LVQ prototype classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic(0x50524F54), d, K, C, then K*C prototypes as float32[d] each."""
import json, struct, sys
import numpy as np

SEED = 1
K_PER_CLASS = 3          # prototypes per class
EPOCHS = 20
LR_START = 0.3
LR_DECAY = 0.95

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
    rng = np.random.default_rng(SEED)

    # Initialize prototypes: K per class, sampled from class members
    protos = np.zeros((K_PER_CLASS * C, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(K_PER_CLASS, len(idx)), replace=False)
        protos[c*K_PER_CLASS:(c+1)*K_PER_CLASS] = X[chosen]

    # LVQ1 training
    n = X.shape[0]
    lr = LR_START
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            target = y[i]
            # squared Euclidean distances to all prototypes
            diff = protos - x
            dists = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dists))
            best_class = best // K_PER_CLASS
            # update closest prototype
            if best_class == target:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])
        lr *= LR_DECAY

    # Write weights.bin
    # magic, d, K, C
    header = struct.pack("<4i", 0x50524F54, d, K_PER_CLASS, C)
    body = protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* LVQ prototype classifier: ./infer weights.bin in.rec N
 * Input record: d bytes (pixel counts). Output: 1 byte class index.
 * Weights format: uint32 magic=0x50524F54, int32 d, K, C; then K*C prototypes of float32[d]. */
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
    int d = hd[1], K = hd[2], C = hd[3];
    int P = K * C;                          // total prototypes
    long expected = 16L + 4L * (long)P * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        int best_proto = 0;
        for (int p = 0; p < P; p++) {
            const float *w = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - w[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_proto = p;
            }
        }
        unsigned char cls = (unsigned char)(best_proto / K);
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```