```hypothesis
Prototype classifier (label-proto) with 4 prototypes per class learned by k-means.
Should beat linear on optdigits because digit classes have multiple modes (writing styles).
Inference: 40 prototypes × 64 dims = 2560 subtractions + squares per item, very fast.
Failure mode: k-means converges poorly or classes aren't prototype-separable.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
K_PER_CLASS = 4
MAX_ITER = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=20):
    n, d = X.shape
    cents = X[:K].copy().astype(np.float32)
    for _ in range(max_iter):
        diff = X[:, None, :] - cents[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_cents[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_cents[k] /= counts[k]
            else:
                new_cents[k] = X[rng.integers(n)]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    
    prototypes = []
    for c in range(C):
        Xc = X_bytes[y == c].astype(np.float32)
        if len(Xc) == 0:
            cents = np.zeros((K_PER_CLASS, d), dtype=np.float32)
        else:
            cents = kmeans(Xc, K_PER_CLASS, rng, MAX_ITER)
        prototypes.append(cents)
    prototypes = np.vstack(prototypes)
    
    magic = 0x544F5250  # "PROT" little-endian
    out = struct.pack("<4i", magic, d, C, K_PER_CLASS)
    out += prototypes.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    int total_proto = C * K;
    long expected = 16 + (long)total_proto * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_f = malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) in_f[j] = xi[j];
        float best_dist = 1e30f;
        int best_class = 0;
        for (int p = 0; p < total_proto; p++) {
            const float *proto = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = in_f[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = p / K;
            }
        }
        unsigned char out = (unsigned char)best_class;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```