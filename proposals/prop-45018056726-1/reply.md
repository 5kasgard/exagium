```hypothesis
Family: label-lvq (Learning Vector Quantization). 
Why: Optdigits has non-linear class boundaries; LVQ learns multiple prototypes per class (3×10=30) capturing local structure better than a single linear boundary, while inference stays fast (30×64 multiply-adds). 
Failure signal: dev accuracy ≤ 0.961 (label-linear's 0.9613) or inference cost ≥ 5000 Ir/item.
```

```python
"""LVQ family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. 3 prototypes/class, 50 epochs, LR 0.1→0.01.
Exports: magic 0x4C565131, d, P, C; float32 mean[d], std[d]; then P×(uint8 class, float32 vec[d])."""
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 3
EPOCHS = 50
LR_INIT, LR_FINAL = 0.1, 0.01

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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score per feature
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # initialize prototypes: 3 random samples per class
    protos, proto_y = [], []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(PROTOS_PER_CLASS, len(idx)), replace=False)
        protos.append(Z[chosen])
        proto_y.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)  # (P, d)
    proto_y = np.concatenate(proto_y)              # (P,)
    P = protos.shape[0]

    # LVQ1 training
    n = Z.shape[0]
    for ep in range(EPOCHS):
        lr = LR_INIT * (LR_FINAL / LR_INIT) ** (ep / (EPOCHS - 1)) if EPOCHS > 1 else LR_INIT
        order = rng.permutation(n)
        for i in order:
            z, target = Z[i], y[i]
            dists = np.sum((protos - z) ** 2, axis=1)
            best = int(np.argmin(dists))
            if proto_y[best] == target:
                protos[best] += lr * (z - protos[best])
            else:
                protos[best] -= lr * (z - protos[best])

    # pack weights
    magic = 0x4C565131  # "LVQ1"
    hdr = struct.pack("<4i", magic, d, P, C)
    body = mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    for c, vec in zip(proto_y, protos):
        body += struct.pack("<B", int(c)) + vec.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```

```c
/* LVQ inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (nearest prototype). */
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
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t*)wb;
    if (hd[0] != 0x4C565131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 8L*d + (long)P * (1L + 4L*d);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float*)(wb + 16);
    const float *sd   = mean + d;
    const unsigned char *pcls = (const unsigned char*)(sd + d);
    const float *pvec = (const float*)(pcls + P);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *znorm = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) znorm[j] = (xi[j] - mean[j]) / sd[j];

        float best_dist = 1e30f; unsigned char best_cls = 0;
        for (int p = 0; p < P; p++) {
            const float *pv = pvec + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) { float diff = znorm[j] - pv[j]; dist += diff * diff; }
            if (dist < best_dist) { best_dist = dist; best_cls = pcls[p]; }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```