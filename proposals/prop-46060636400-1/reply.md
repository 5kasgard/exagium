```hypothesis
Family: label-proto (prototype-based nearest-neighbor with learned prototypes per class).
Why: 8x8 digit classes have non-linear boundaries; a small set of prototypes (3-5/class) captures within-class variation better than a linear model while keeping inference fast (~30 prototypes * 64 dims = 1920 FLOPs/item vs linear's 640 adds). Prototypes are learned by per-class k-means++ on z-scored pixels (deterministic seed), with P_per_class tuned on the last 20% of train.in.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost > 3× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_per_class(X, y, C, P_per_class, rng, max_iter=20):
    d = X.shape[1]
    P_total = P_per_class * C
    prototypes = np.zeros((P_total, d), dtype=np.float32)
    proto_class = np.zeros(P_total, dtype=np.uint8)
    for c in range(C):
        Xc = X[y == c]
        n_c = len(Xc)
        if n_c == 0:
            continue
        P_use = min(P_per_class, n_c)
        centroids = np.zeros((P_use, d), dtype=np.float32)
        idx = rng.integers(n_c)
        centroids[0] = Xc[idx]
        for k in range(1, P_use):
            dist2 = np.full(n_c, np.inf, dtype=np.float32)
            for j in range(k):
                diff = Xc - centroids[j]
                d2 = np.sum(diff * diff, axis=1)
                dist2 = np.minimum(dist2, d2)
            probs = dist2 / dist2.sum()
            idx = rng.choice(n_c, p=probs)
            centroids[k] = Xc[idx]
        for _ in range(max_iter):
            dists = np.zeros((n_c, P_use), dtype=np.float32)
            for k in range(P_use):
                diff = Xc - centroids[k]
                dists[:, k] = np.sum(diff * diff, axis=1)
            assign = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(P_use, dtype=np.int32)
            for i in range(n_c):
                k = assign[i]
                new_centroids[k] += Xc[i]
                counts[k] += 1
            for k in range(P_use):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]
                else:
                    idx = rng.integers(n_c)
                    new_centroids[k] = Xc[idx]
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        start = c * P_per_class
        prototypes[start:start + P_use] = centroids
        proto_class[start:start + P_use] = c
    return prototypes, proto_class


def evaluate(X, y, prototypes, proto_class, proto_norm2):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        z = X[i]
        z_norm2 = np.dot(z, z)
        best_dist = np.inf
        best_cls = 0
        for p_idx, p in enumerate(prototypes):
            dot = np.dot(z, p)
            dist2 = z_norm2 + proto_norm2[p_idx] - 2.0 * dot
            if dist2 < best_dist:
                best_dist = dist2
                best_cls = proto_class[p_idx]
        if best_cls == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    mu = X_raw.mean(axis=0).astype(np.float32)
    sd = X_raw.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_raw - mu) / sd).astype(np.float32)

    n = len(X)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_P = 1
    best_acc = -1.0
    for P_per_class in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(SEED + P_per_class)
        prototypes, proto_class = kmeans_per_class(X_train, y_train, C, P_per_class, rng)
        proto_norm2 = np.sum(prototypes * prototypes, axis=1)
        acc = evaluate(X_tune, y_tune, prototypes, proto_class, proto_norm2)
        if acc > best_acc:
            best_acc = acc
            best_P = P_per_class

    rng = np.random.default_rng(SEED)
    prototypes, proto_class = kmeans_per_class(X, y, C, best_P, rng)
    proto_norm2 = np.sum(prototypes * prototypes, axis=1)
    P_total = best_P * C

    out = struct.pack("<4i", 0x544F5250, d, best_P, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += prototypes.astype("<f4").tobytes()
    out += proto_class.astype("<u1").tobytes()
    out += proto_norm2.astype("<f4").tobytes()

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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P_per_class = hd[2], C = hd[3];
    int P_total = P_per_class * C;
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights params\n"); return 3; }

    size_t offset = 16;
    float *mean = (float *)(wb + offset); offset += d * sizeof(float);
    float *std = (float *)(wb + offset); offset += d * sizeof(float);
    float *prototypes = (float *)(wb + offset); offset += (long)P_total * d * sizeof(float);
    unsigned char *proto_class = wb + offset; offset += P_total;
    float *proto_norm2 = (float *)(wb + offset); offset += P_total * sizeof(float);

    if (nw != (long)offset) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float z_norm2 = 0.0f;
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
            z_norm2 += z[j] * z[j];
        }

        float best_dist2 = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < P_total; p++) {
            float dot = 0.0f;
            const float *proto = prototypes + (long)p * d;
            for (int j = 0; j < d; j++) {
                dot += z[j] * proto[j];
            }
            float dist2 = z_norm2 + proto_norm2[p] - 2.0f * dot;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_class = proto_class[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```