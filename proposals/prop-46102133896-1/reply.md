```hypothesis
label-proto: k-means prototypes per class (3 per digit = 30 total). Captures multi-modal digit shapes better than a single linear boundary. Inference: 30×64 byte-wise squared Euclidean distances (integer arithmetic, no tables). Fails if dev accuracy < 0.9613 (label-linear) or Ir/item > 10000.
```
```python
"""label-proto: prototype classifier with k-means per class.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys

import numpy as np

SEED = 1
K_PER_CLASS = 3
MAX_ITER = 20
MAGIC = 0x50524F54  # 'PROT'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(Xc, k, rng):
    """Deterministic k-means on uint8 data, returns k centroids as uint8."""
    n = Xc.shape[0]
    if n <= k:
        # Pad by repeating if needed (should not happen for optdigits)
        centroids = Xc[:k].copy()
        if n < k:
            centroids = np.vstack([centroids, np.tile(Xc[0], (k - n, 1))])
        return centroids.astype(np.uint8)
    # Random initial centroids from data (without replacement)
    idx = rng.choice(n, size=k, replace=False)
    centroids = Xc[idx].astype(np.float64)
    for _ in range(MAX_ITER):
        # Assign: squared Euclidean distance
        diff = Xc[:, None, :] - centroids[None, :, :]  # (n, k, d)
        dist2 = np.sum(diff * diff, axis=2)            # (n, k)
        assign = np.argmin(dist2, axis=1)              # (n,)
        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for j in range(n):
            c = assign[j]
            new_centroids[c] += Xc[j]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                # Empty cluster: re-initialize to random data point
                new_centroids[c] = Xc[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return np.round(centroids).astype(np.uint8)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    all_protos = []
    all_labels = []
    for c in range(C):
        Xc = X[y == c]
        protos = kmeans_class(Xc, K_PER_CLASS, rng)
        all_protos.append(protos)
        all_labels.append(np.full(K_PER_CLASS, c, dtype=np.uint8))

    protos = np.vstack(all_protos)      # (K, d)
    labels = np.concatenate(all_labels) # (K,)
    K = protos.shape[0]

    # Write weights: magic, d, K, C, then protos (K*d bytes), then labels (K bytes)
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += protos.tobytes()
    out += labels.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Weights: magic(4), d(4), K(4), C(4), protos[K][d] (uint8), labels[K] (uint8).
 * Input record: d bytes. Output: 1 byte class of nearest prototype (squared L2). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || K <= 0 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + (long)K * d + K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const unsigned char *protos = (const unsigned char *)(hd + 4);
    const unsigned char *plabels = protos + (long)K * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int32_t best_dist = 0x7fffffff;
        unsigned char best_label = 0;
        for (int p = 0; p < K; p++) {
            const unsigned char *cent = protos + (long)p * d;
            int32_t dist = 0;
            for (int j = 0; j < d; j++) {
                int diff = (int)xi[j] - (int)cent[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_label = plabels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```