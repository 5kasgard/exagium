```hypothesis
Family: label-lvq (Learning Vector Quantization with 3 prototypes/class).
Why: Optdigits has 64 ordinal pixels (0..16); LVQ learns class prototypes in z-scored space, capturing non-linear boundaries that linear models miss, while inference is just 30×64 squared distances (≈2k FLOPs) – far cheaper than MLP/GBDT.
Failure signal: dev accuracy ≤ 0.962 or instructions/item > 15000.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
N_PROTO_PER_CLASS = 3
EPOCHS = 50
LR_START = 0.1
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # Per-pixel z-score
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    # Deterministic prototype init: first 3 samples of each class
    prototypes, proto_labels = [], []
    for c in range(C):
        idx = np.where(y == c)[0]
        for i in idx[:N_PROTO_PER_CLASS]:
            prototypes.append(Xz[i].copy())
            proto_labels.append(c)
    prototypes = np.array(prototypes, dtype=np.float32)   # (n_proto, d)
    proto_labels = np.array(proto_labels, dtype=np.uint8)
    n_proto = len(prototypes)

    # LVQ1 training
    rng = np.random.default_rng(SEED)
    n = Xz.shape[0]
    for epoch in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / (EPOCHS - 1)) if EPOCHS > 1 else LR_START
        order = rng.permutation(n)
        for i in order:
            x = Xz[i]
            yi = y[i]
            diffs = prototypes - x
            dists = np.sum(diffs * diffs, axis=1)
            best = int(np.argmin(dists))
            if proto_labels[best] == yi:
                prototypes[best] += lr * (x - prototypes[best])
            else:
                prototypes[best] -= lr * (x - prototypes[best])

    # Write weights: magic, d, n_proto, C, mean[d], std[d], proto[n_proto][d], labels[n_proto]
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, n_proto, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += prototypes.astype("<f4").tobytes()
    out += proto_labels.astype("<u1").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d != 64 || C != 10 || n_proto != 30) { fprintf(stderr, "unexpected dims\n"); return 3; }

    float *mean = (float*)(hd + 4);
    float *std  = mean + d;
    float *proto = std + d;
    unsigned char *plabels = (unsigned char*)(proto + (long)n_proto * d);
    long expected = 16 + (long)d*4 + (long)d*4 + (long)n_proto*d*4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(d * sizeof(float));
    float *dists = malloc(n_proto * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mean[j]) / std[j];
        for (int p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *pr = proto + (long)p * d;
            for (int j = 0; j < d; j++) { float diff = xz[j] - pr[j]; sum += diff * diff; }
            dists[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < n_proto; p++) if (dists[p] < dists[best]) best = p;
        unsigned char pred = plabels[best];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```