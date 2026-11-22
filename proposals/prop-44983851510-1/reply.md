```hypothesis
Family: label-lvq (Learning Vector Quantization with 3 prototypes per class).
Why: Captures non-linear boundaries via multiple prototypes per class; 8x8 digit shapes have local variations that single linear decision boundaries miss. LVQ is fast at inference (30 * 64 FLOPs) and deterministic with fixed seed.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear significantly.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 3
EPOCHS = 20
ALPHA_START = 0.1
ALPHA_END = 0.001

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
    d = card["input"]["bytes"]
    P = PROTOS_PER_CLASS * C
    
    rng = np.random.default_rng(SEED)
    
    # Initialize prototypes: for each class, pick PROTOS_PER_CLASS random samples
    protos = np.zeros((P, d), dtype=np.float32)
    proto_labels = np.zeros(P, dtype=np.uint8)
    idx = 0
    for c in range(C):
        class_mask = (y == c)
        class_samples = X[class_mask]
        if len(class_samples) == 0:
            continue
        chosen = rng.choice(len(class_samples), size=PROTOS_PER_CLASS, replace=False)
        protos[idx:idx+PROTOS_PER_CLASS] = class_samples[chosen].astype(np.float32)
        proto_labels[idx:idx+PROTOS_PER_CLASS] = c
        idx += PROTOS_PER_CLASS
    
    # LVQ training
    n = X.shape[0]
    X_float = X.astype(np.float32)
    for epoch in range(EPOCHS):
        alpha = ALPHA_START * (ALPHA_END / ALPHA_START) ** (epoch / (EPOCHS - 1))
        order = rng.permutation(n)
        for i in order:
            x = X_float[i]
            label = y[i]
            diff = protos - x
            dist = np.sum(diff * diff, axis=1)
            nearest = np.argmin(dist)
            if proto_labels[nearest] == label:
                protos[nearest] += alpha * (x - protos[nearest])
            else:
                protos[nearest] -= alpha * (x - protos[nearest])
    
    # Write weights
    magic = 0x4C565131
    out = struct.pack("<4i", magic, d, P, C)
    out += protos.astype("<f4").tobytes()
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
    if (d != 64 || C != 10 || P != 30 || nw != 16 + 4L * d * P + P) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *protos = (const float *)(hd + 4);
    const unsigned char *proto_labels = (const unsigned char *)(protos + (long)d * P);
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        
        for (int p = 0; p < P; p++) {
            float dist = 0.0f;
            const float *proto = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_labels[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```