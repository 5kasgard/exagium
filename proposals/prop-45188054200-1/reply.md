```hypothesis
Family: label-lvq (Learning Vector Quantization). It learns 2 prototypes per class (20 total) in z-scored pixel space using LVQ1. For optdigits, prototypes capture non-linear stroke variations better than a single linear separator per class. Inference computes z-scored input once, then 20 dot products (64-d) and picks the highest score — about 2× the work of label-linear. If dev accuracy ≤ 0.961 or Ir/item ≥ 10000, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
PROTOS_PER_CLASS = 2
LVQ_EPOCHS = 20
LVQ_ALPHA_START = 0.1
LVQ_ALPHA_END = 0.001
KMEANS_ITERS = 10


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

    rng = np.random.default_rng(SEED)

    # Z-score ordinal features (all features are ordinal for optdigits)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    inv_sd = (1.0 / sd).astype(np.float32)
    Z = ((X.astype(np.float32) - mu) * inv_sd).astype(np.float32)

    n = Z.shape[0]
    P = PROTOS_PER_CLASS * C

    # Initialize prototypes: k-means per class
    protos = np.zeros((P, d), dtype=np.float32)
    proto_class = np.zeros(P, dtype=np.uint8)

    p_idx = 0
    for c in range(C):
        Zc = Z[y == c]
        if len(Zc) == 0:
            protos[p_idx:p_idx + PROTOS_PER_CLASS] = rng.normal(0, 1, (PROTOS_PER_CLASS, d)).astype(np.float32)
        elif len(Zc) <= PROTOS_PER_CLASS:
            protos[p_idx:p_idx + len(Zc)] = Zc
            if len(Zc) < PROTOS_PER_CLASS:
                protos[p_idx + len(Zc):p_idx + PROTOS_PER_CLASS] = rng.normal(0, 1, (PROTOS_PER_CLASS - len(Zc), d)).astype(np.float32)
        else:
            idx = rng.choice(len(Zc), PROTOS_PER_CLASS, replace=False)
            centroids = Zc[idx].copy()
            for _ in range(KMEANS_ITERS):
                dists = np.sum((Zc[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                assign = np.argmin(dists, axis=1)
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(PROTOS_PER_CLASS, dtype=np.int32)
                for i, a in enumerate(assign):
                    new_centroids[a] += Zc[i]
                    counts[a] += 1
                for k in range(PROTOS_PER_CLASS):
                    if counts[k] > 0:
                        new_centroids[k] /= counts[k]
                    else:
                        new_centroids[k] = rng.normal(0, 1, d).astype(np.float32)
                centroids = new_centroids
            protos[p_idx:p_idx + PROTOS_PER_CLASS] = centroids
        proto_class[p_idx:p_idx + PROTOS_PER_CLASS] = c
        p_idx += PROTOS_PER_CLASS

    # LVQ1 training
    alpha = LVQ_ALPHA_START
    decay = (LVQ_ALPHA_END / LVQ_ALPHA_START) ** (1.0 / LVQ_EPOCHS)

    for _ in range(LVQ_EPOCHS):
        order = rng.permutation(n)
        for i in order:
            z = Z[i]
            target = y[i]
            dists = np.sum((protos - z) ** 2, axis=1)
            best = int(np.argmin(dists))
            if proto_class[best] == target:
                protos[best] += alpha * (z - protos[best])
            else:
                protos[best] -= alpha * (z - protos[best])
        alpha *= decay

    proto_norm_sq = np.sum(protos ** 2, axis=1).astype(np.float32)

    # Export: magic, d, P, C; mu[d], inv_sd[d], protos[P*d], proto_norm_sq[P], proto_class[P]
    out = struct.pack("<4i", 0x4C565131, d, P, C)
    out += mu.astype("<f4").tobytes()
    out += inv_sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_norm_sq.astype("<f4").tobytes()
    out += proto_class.astype("<u1").tobytes()

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
    if (d != 64 || C != 10 || P <= 0 || P > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 4L*d + 4L*d + 4L*P*d + 4L*P + P;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *inv_sd = mu + d;
    const float *protos = inv_sd + d;
    const float *proto_norm_sq = protos + (long)P * d;
    const unsigned char *proto_class = (const unsigned char *)(proto_norm_sq + P);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * P);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) * inv_sd[j];
        }

        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            const float *proto = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                sum += z[j] * proto[j];
            }
            scores[p] = sum - 0.5f * proto_norm_sq[p];
        }

        unsigned char best_class = proto_class[0];
        float best_score = scores[0];
        for (int p = 1; p < P; p++) {
            if (scores[p] > best_score) {
                best_score = scores[p];
                best_class = proto_class[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```