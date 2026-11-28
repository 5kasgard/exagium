```hypothesis
Family: label-lvq (Learning Vector Quantization). 
Why: LVQ learns a small set of prototypes per class (e.g., 3) in the normalized pixel space, giving non-linear decision boundaries at inference cost of only P*C*d multiply-adds (30*64=1920). On 8x8 digits this should exceed linear's 96.1% while staying well below MLP cost. 
Failure signal: dev accuracy ≤ 0.961 (linear's mean) or Ir/item > 8000.
```

```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports: magic 0x4C565131, d, C, P, float32 mean[d], std[d], prototypes[P*C][d]."""
import json, math, struct, sys
import numpy as np

SEED = 1
P = 3                 # prototypes per class
MAX_EPOCHS = 50
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
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score per pixel
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    # initialize prototypes: pick P random examples per class
    protos = np.zeros((P * C, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        choice = rng.choice(idx, size=P, replace=False)
        protos[c * P:(c + 1) * P] = Xn[choice]

    n = Xn.shape[0]
    for epoch in range(MAX_EPOCHS):
        lr = INIT_LR * (1.0 - epoch / MAX_EPOCHS)
        order = rng.permutation(n)
        for i in order:
            x = Xn[i]
            # squared Euclidean distances to all prototypes
            diff = protos - x
            dist2 = (diff * diff).sum(axis=1)
            j = int(np.argmin(dist2))
            c_proto = j // P
            if c_proto == y[i]:
                protos[j] += lr * (x - protos[j])
            else:
                protos[j] -= lr * (x - protos[j])

    # pack weights
    out = struct.pack("<4i", 0x4C565131, d, C, P)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte class.
 * Weights format: magic(0x4C565131), d, C, P, mean[d], std[d], prototypes[P*C][d] (all float32 LE). */
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
    uint8_t *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], P = hd[3];
    if (d <= 0 || C <= 0 || P <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * (long)P * C * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *protos = std + d;           // [P*C][d]
    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;

    float *xn = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xn[j] = (xi[j] - mean[j]) / std[j];

        float best_dist = 1e30f; int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float dist2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xn[j] - proto[j];
                    dist2 += diff * diff;
                }
                if (dist2 < best_dist) { best_dist = dist2; best_class = c; }
            }
        }
        unsigned char out = (unsigned char)best_class;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```