```hypothesis
Family: label-proto (prototype-based classifier with LVQ refinement).
Why: Optdigits forms tight per-digit clusters in pixel space; a small set of learned prototypes (3 per class = 30 total) captures non-linear boundaries better than a linear model while inference requires only ~30×64 ≈ 2k simple ops (squared Euclidean distances), far cheaper than an MLP.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 5000.
```
```python
"""label-proto: Prototype-based classifier (LVQ) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread. Learns 3 prototypes per class via k-means++
initialization followed by LVQ1 refinement. Exports prototypes as float32.
Weights format: magic(0x50524F54), d, n_proto, C; proto[d][n_proto] (float32); proto_class[n_proto] (uint8)."""
import json, math, struct, sys

import numpy as np

PROTO_PER_CLASS = 3
SEED = 1
LVQ_EPOCHS = 15
LVQ_LR_START = 0.3
LVQ_LR_END = 0.01


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_plusplus(X, k, rng):
    """X: (n, d) float32. Return (k, d) centroids."""
    n, d = X.shape
    cents = np.empty((k, d), dtype=np.float32)
    cents[0] = X[rng.integers(n)]
    for i in range(1, k):
        dist2 = np.min(np.sum((X[:, None, :] - cents[:i]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        cents[i] = X[rng.choice(n, p=probs)]
    return cents


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    Xf = X.astype(np.float32)
    n_proto = PROTO_PER_CLASS * C
    protos = np.empty((n_proto, d), dtype=np.float32)
    proto_class = np.empty(n_proto, dtype=np.uint8)

    # Initialize: k-means++ per class
    for c in range(C):
        Xc = Xf[y == c]
        if len(Xc) == 0:
            # Fallback: random from all data
            cents = Xf[rng.choice(len(Xf), PROTO_PER_CLASS, replace=False)]
        elif len(Xc) <= PROTO_PER_CLASS:
            cents = Xc[:PROTO_PER_CLASS]
            if len(cents) < PROTO_PER_CLASS:
                cents = np.vstack([cents, Xf[rng.choice(len(Xf), PROTO_PER_CLASS - len(cents), replace=False)]])
        else:
            cents = kmeans_plusplus(Xc, PROTO_PER_CLASS, rng)
        base = c * PROTO_PER_CLASS
        protos[base:base + PROTO_PER_CLASS] = cents
        proto_class[base:base + PROTO_PER_CLASS] = c

    # LVQ1 refinement
    n = len(Xf)
    for epoch in range(LVQ_EPOCHS):
        lr = LVQ_LR_START * (LVQ_LR_END / LVQ_LR_START) ** (epoch / max(1, LVQ_EPOCHS - 1))
        order = rng.permutation(n)
        for idx in order:
            x = Xf[idx]
            target = y[idx]
            # squared Euclidean distances to all prototypes
            dist2 = np.sum((protos - x) ** 2, axis=1)
            best = np.argmin(dist2)
            if proto_class[best] == target:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # Pack weights
    # magic, d, n_proto, C
    hdr = struct.pack("<4i", 0x50524F54, d, n_proto, C)
    body = protos.astype("<f4").tobytes() + proto_class.astype("<B").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output: one byte class (nearest prototype, squared L2).
 * Weights: magic(0x50524F54), d, n_proto, C; float32 protos[n_proto][d]; uint8 proto_class[n_proto]. */
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
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d <= 0 || n_proto <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)n_proto * d * 4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *protos = (const float *)(wb + 16);
    const unsigned char *proto_class = wb + 16 + (long)n_proto * d * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *dist2 = malloc(sizeof(float) * n_proto);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *pr = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xi[j] - pr[j];
                sum += diff * diff;
            }
            dist2[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < n_proto; p++)
            if (dist2[p] < dist2[best]) best = p;
        unsigned char pred = proto_class[best];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```