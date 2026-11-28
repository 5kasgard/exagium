```hypothesis
Family: label-avgperceptron (Averaged Perceptron). The optdigits pixels are ordinal 0..16; z-scoring gives a dense linear problem where the averaged perceptron often matches or beats logistic regression (label-linear) because its implicit margin maximization generalizes better on small datasets. Inference is a single dot product per class — same O(d·C) cost as label-linear. If dev accuracy ≤ 0.960 or Ir/item > 5500, the hypothesis fails.
```

```python
"""Averaged Perceptron for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encoding: ordinal z-scored per byte (same as label-linear).
Exported weights: magic 0x41564750, d, C; float32 W[d][C], b[C] (averaged weights)."""
import json, struct, sys

import numpy as np

SEED, EPOCHS = 1, 20


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

    # Encode: ordinal -> z-scored per column
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1)
    n, k = Z.shape

    rng = np.random.default_rng(SEED)
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    W_avg = np.zeros((k, C), np.float32)
    b_avg = np.zeros(C, np.float32)
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for i in order:
            z = Z[i]
            scores = z @ W + b
            pred = int(np.argmax(scores))
            true = int(y[i])
            if pred != true:
                W[:, true] += z
                W[:, pred] -= z
                b[true] += 1.0
                b[pred] -= 1.0
            t += 1
            W_avg += W
            b_avg += b

    W_avg /= t
    b_avg /= t

    out = struct.pack("<3i", 0x41564750, d, C)
    out += W_avg.astype("<f4").tobytes()
    out += b_avg.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Averaged Perceptron for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x41564750, d, C; float32 W[d][C], b[C] (row-major W). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 12 || hd[0] != 0x41564750) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 12 + 4L * ((long)d * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *wj = W + (long)j * C;
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) scores[c] += wj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```