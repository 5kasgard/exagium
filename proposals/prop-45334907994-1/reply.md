```hypothesis
Family: label-proto (LVQ prototype classifier). Optdigits has 10 classes with non-linear boundaries in 64D pixel space; a small set of learned prototypes per class (k=3, 30 total) captures local structure better than a single linear separator while keeping inference to ~30×64 distance computations. Fixed seed, deterministic LVQ on z-scored features. If dev accuracy ≤ 0.9613 (label-linear), the prototype approach does not add value here.
```

```python
"""label-proto: LVQ prototype classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1). Per-feature z-scoring on train. k=3 prototypes/class via k-means++ init,
then LVQ1 for 20 epochs (lr 0.1 -> 0.01). Exports magic, d, n_proto, C, mean[d], std[d],
prototypes[n_proto][d] (float32), proto_class[n_proto] (uint8)."""
import json, math, struct, sys

import numpy as np

K_PER_CLASS = 3
SEED, LVQ_EPOCHS, LR_START, LR_END = 1, 20, 0.1, 0.01


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_plusplus_init(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for i in range(1, k):
        dist2 = np.min(np.sum((X[:, None, :] - centroids[:i]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    return centroids


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score per feature on train
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    # initialize prototypes per class
    protos = []
    proto_cls = []
    for c in range(C):
        Xc = Xn[y == c]
        if len(Xc) == 0:
            continue
        k = min(K_PER_CLASS, len(Xc))
        centroids = kmeans_plusplus_init(Xc, k, rng)
        protos.append(centroids)
        proto_cls.extend([c] * k)
    protos = np.vstack(protos).astype(np.float32)  # [n_proto, d]
    proto_cls = np.array(proto_cls, dtype=np.uint8)
    n_proto = protos.shape[0]

    # LVQ1
    for epoch in range(LVQ_EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / max(1, LVQ_EPOCHS - 1))
        order = rng.permutation(len(Xn))
        for idx in order:
            x = Xn[idx]
            target = y[idx]
            # squared Euclidean distances
            dist2 = np.sum((protos - x) ** 2, axis=1)
            best = int(np.argmin(dist2))
            best_cls = proto_cls[best]
            if best_cls == target:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # pack weights: magic, d, n_proto, C, mean[d], std[d], protos[n_proto*d], proto_cls[n_proto]
    magic = 0x50524F54  # "PROT"
    out = struct.pack("<4i", magic, d, n_proto, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_cls.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Loads z-score params and prototypes, computes squared Euclidean distance,
 * outputs class of nearest prototype (first min wins). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d <= 0 || n_proto <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * n_proto * d + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mu = (const float *)(wb + 16);
    const float *sd = mu + d;
    const float *protos = sd + d;
    const unsigned char *proto_cls = (const unsigned char *)(protos + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xn = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xn[j] = (xi[j] - mu[j]) / sd[j];
        float best_dist = 1e30f; unsigned char best_cls = 0;
        for (int p = 0; p < n_proto; p++) {
            const float *pr = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = xn[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_cls = proto_cls[p];
            }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```