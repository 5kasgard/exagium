```hypothesis
Family: label-proto (prototype-based classifier). Learns 3 prototypes per class via k-means on training data (30 prototypes total, 64-d uint8). Inference computes squared Euclidean distance to all prototypes and picks the nearest; extremely fast (≈30×64 ops) and competitive with k-NN accuracy since prototypes summarize class manifolds. Fails if dev accuracy < 0.965 or Ir/item > 6000.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 3
MAX_ITER = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, k, rng, max_iter=100):
    n, d = X.shape
    centroids = X[rng.choice(n, k, replace=False)].astype(np.float32)
    for _ in range(max_iter):
        diff = X[:, None, :] - centroids[None, :, :]
        dist2 = (diff * diff).sum(axis=2)
        assign = dist2.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, np.int32)
        for i in range(n):
            c = assign[i]
            new_centroids[c] += X[i]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    prototypes = []
    proto_labels = []
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        centroids = kmeans(Xc.astype(np.float32), PROTOS_PER_CLASS, rng, MAX_ITER)
        prototypes.append(np.clip(np.rint(centroids), 0, 255).astype(np.uint8))
        proto_labels.append(np.full(PROTOS_PER_CLASS, c, dtype=np.uint8))

    P = np.vstack(prototypes)
    L = np.concatenate(proto_labels)
    P_count = len(P)

    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, P_count, C)
    out += P.tobytes()
    out += L.tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    long expected = 16L + (long)P * d + P;
    if (nw != expected || d != 64 || C != 10 || P <= 0) { fprintf(stderr, "bad weights size\n"); return 3; }
    const unsigned char *protos = (const unsigned char *)(hd + 4);
    const unsigned char *plabels = protos + (long)P * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char best_label = 0;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        unsigned best = 0;
        unsigned best_dist = ~0u;
        for (int p = 0; p < P; p++) {
            const unsigned char *proto = protos + (long)p * d;
            unsigned dist = 0;
            for (int j = 0; j < d; j++) {
                int diff = (int)xi[j] - (int)proto[j];
                dist += (unsigned)(diff * diff);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = plabels[p];
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```