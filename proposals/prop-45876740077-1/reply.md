```hypothesis
Family: label-proto (prototype-based classifier). Learns 3 prototypes per class via per-class k-means
(deterministic seed). At inference, assigns each sample to the class of its nearest prototype (Euclidean
distance on raw 0..16 pixels). Exploits the multi-modal shape of each digit (e.g. open/closed 4) with
far fewer distance computations than k-NN (30 vs 2867). Expect ~0.965 accuracy at ~2000 Ir/item.
Failure: dev accuracy ≤ 0.960 (prototypes too crude for this task).
```
```python
"""label-proto: per-class k-means prototypes for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

SEED = 1
K_PER_CLASS = 3          # prototypes per class
MAX_ITER = 100           # k-means iterations


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(Xc, K, rng):
    """Deterministic k-means++ initialization + Lloyd, returns (K, d) centroids."""
    n, d = Xc.shape
    if n <= K:
        # pad with duplicates if very few samples
        cents = Xc.astype(np.float32)
        if n < K:
            extra = np.tile(cents[:1], (K - n, 1))
            cents = np.vstack([cents, extra])
        return cents
    # k-means++ init
    cents = np.empty((K, d), np.float32)
    cents[0] = Xc[rng.integers(n)]
    for k in range(1, K):
        dist2 = np.min(np.sum((Xc[:, None] - cents[:k]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        cents[k] = Xc[rng.choice(n, p=probs)]
    # Lloyd iterations
    for _ in range(MAX_ITER):
        dist2 = np.sum((Xc[:, None] - cents) ** 2, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, np.int32)
        for i in range(n):
            c = assign[i]
            new_cents[c] += Xc[i]
            counts[c] += 1
        for k in range(K):
            if counts[k]:
                new_cents[k] /= counts[k]
            else:
                # reassign empty cluster to farthest point from its centroid
                far = np.argmax(np.min(np.sum((Xc[:, None] - cents) ** 2, axis=2), axis=1))
                new_cents[k] = Xc[far]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Learn prototypes per class
    proto = np.zeros((C, K_PER_CLASS, d), np.float32)
    for c in range(C):
        Xc = X[y == c].astype(np.float32)
        proto[c] = kmeans_class(Xc, K_PER_CLASS, rng)

    # Flatten to (C*K, d)
    proto_flat = proto.reshape(C * K_PER_CLASS, d)

    # Write weights: magic, d, K, C, then proto_flat as float32 LE
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, K_PER_CLASS, C)
    out += proto_flat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Weights: magic 0x50524F54, int32 d, K, C; then (C*K*d) float32 prototypes.
 * Each input record is d bytes (0..16). Output: class of nearest prototype (squared Euclidean). */
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
    if (d <= 0 || K <= 0 || C <= 0 || C > 256 ||
        nw != 16 + 4L * (long)C * K * d) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *proto = (const float *)(hd + 4);           /* [C*K][d] */
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_dist = 1e30f;
        unsigned char best_class = 0;

        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *p = proto + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float delta = (float)xi[j] - p[j];
                    dist += delta * delta;
                    if (dist >= best_dist) break;  /* early exit */
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```