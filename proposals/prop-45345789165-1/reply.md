```hypothesis
label-proto: Prototype-based classifier (condensed k-NN) with k=3 prototypes per class learned by k-means. Exploits non-linear manifold structure of handwritten digits with compute similar to linear model (fixed small prototype set). Fails if dev accuracy <= 0.9613 or Ir/item > 5058.
```
```python
import json, struct, sys

import numpy as np

MAGIC = 0x544F5250  # 'PROT'
SEED = 1
K_PER_CLASS = 3
MAX_ITER = 100
TOL = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_pp(X, k, rng):
    n, d = X.shape
    centers = np.zeros((k, d), dtype=np.float64)
    idx = rng.integers(n)
    centers[0] = X[idx]
    dist2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        d2 = np.sum((X - centers[i - 1]) ** 2, axis=1)
        np.minimum(dist2, d2, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    return centers


def kmeans(X, k, rng, max_iter=100, tol=1e-4):
    n, d = X.shape
    centers = kmeans_pp(X, k, rng)
    for _ in range(max_iter):
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = labels[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centers[j] /= counts[j]
            else:
                new_centers[j] = X[rng.integers(n)]
        shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)).max()
        centers = new_centers
        if shift < tol:
            break
    return centers


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    all_prototypes = []
    all_proto_classes = []

    for c in range(C):
        Xc = X_bytes[y == c].astype(np.float64)
        if len(Xc) == 0:
            continue
        k = min(K_PER_CLASS, len(Xc))
        centers = kmeans(Xc, k, rng, MAX_ITER, TOL)
        all_prototypes.append(centers)
        all_proto_classes.extend([c] * k)

    prototypes = np.vstack(all_prototypes).astype(np.float32)
    proto_classes = np.array(all_proto_classes, dtype=np.uint8)
    n_proto = len(proto_classes)

    out = struct.pack("<4i", MAGIC, d, n_proto, C)
    out += prototypes.tobytes()
    out += proto_classes.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x544F5250) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d <= 0 || n_proto <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)n_proto * d * 4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *prototypes = (float *)(wb + 16);
    unsigned char *proto_classes = wb + 16 + (long)n_proto * d * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < n_proto; p++) {
            const float *proto = prototypes + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_classes[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```