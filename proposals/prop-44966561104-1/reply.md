```hypothesis
Family: label-proto (Learning Vector Quantization / prototype classifier).
Why: Optdigits has 64 ordinal pixel features (0–16) where each digit class forms multiple compact clusters in pixel space. A linear model (label-linear) cannot capture non-linear boundaries; k-NN (label-knn) is accurate but slow (2.2M Ir/item). A small set of learned prototypes per class (e.g., 4 per digit = 40 total) gives non-linear decision boundaries with O(prototypes × dims) inference cost (~40×64 = 2560 ops), far cheaper than k-NN. Training uses deterministic per-class k-means on the training split.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 10000.
```

```python
"""label-proto: prototype classifier (per-class k-means) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic(0x50524F54), nproto, dims; float32 prototypes[nproto][dims]; uint8 labels[nproto]."""
import json, struct, sys

import numpy as np

SEED = 1
K_PER_CLASS = 4
MAX_ITER = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng):
    """X: (n, d) float32. Return (K, d) float32 centroids."""
    n, d = X.shape
    if n <= K:
        return X.copy()
    idx = rng.choice(n, K, replace=False)
    cents = X[idx].copy()
    for _ in range(MAX_ITER):
        # distances (n, K)
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    Xf = X.astype(np.float32)
    all_cents = []
    all_labels = []
    for c in range(C):
        Xc = Xf[y == c]
        cents = kmeans(Xc, K_PER_CLASS, rng)
        all_cents.append(cents)
        all_labels.append(np.full(len(cents), c, dtype=np.uint8))

    cents = np.vstack(all_cents).astype(np.float32)  # (nproto, d)
    labels = np.concatenate(all_labels).astype(np.uint8)  # (nproto,)
    nproto = cents.shape[0]

    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<3i", magic, nproto, d)
    out += cents.tobytes()
    out += labels.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output: 1 byte class label.
 * Weights: magic(0x50524F54), nproto(int32), dims(int32),
 *          float32 prototypes[nproto][dims], uint8 labels[nproto]. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int nproto = hd[1], d = hd[2];
    if (nproto <= 0 || d <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 12L + (long)nproto * d * 4L + nproto;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *protos = (const float *)(hd + 3);
    const unsigned char *plabels = (const unsigned char *)(protos + (long)nproto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *buf = (float *)malloc(d * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) buf[j] = (float)xi[j];

        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < nproto; p++) {
            const float *cent = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = buf[j] - cent[j];
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