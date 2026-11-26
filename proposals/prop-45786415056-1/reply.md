```hypothesis
Family: label-proto (per-class k-means prototypes). Optdigits digits have multiple writing styles per class; prototypes capture multi-modal densities better than linear boundaries. Inference is O(C*K*d) distance computations (no matmul), so it should match label-mlp16 accuracy (~0.977) at lower cost than label-linear. Failure if dev accuracy < 0.965 or Ir/item > 6000.
```
```python
import json, struct, sys, math
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    centroids = np.zeros((K, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for k in range(1, K):
        dists = np.min(np.sum((X[:, None] - centroids[:k])**2, axis=2), axis=1)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
    for _ in range(max_iter):
        dists = np.sum((X[:, None] - centroids)**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                new_centroids[k] = X[rng.integers(n)]
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

    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    n = len(Xz)
    split = int(n * 0.8)
    X_train, X_tune = Xz[:split], Xz[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_K, best_acc = 1, -1.0
    for K in [1, 2, 3, 4, 5, 6, 8, 10]:
        protos = np.zeros((C, K, d), dtype=np.float32)
        for c in range(C):
            Xc = X_train[y_train == c]
            if len(Xc) == 0:
                protos[c] = 0
            elif len(Xc) <= K:
                protos[c, :len(Xc)] = Xc
                if len(Xc) < K:
                    protos[c, len(Xc):] = Xc[rng.integers(len(Xc), size=K - len(Xc))]
            else:
                protos[c] = kmeans(Xc, K, rng)
        correct = 0
        for i in range(len(X_tune)):
            xi = X_tune[i]
            best_dist = 1e30
            best_c = 0
            for c in range(C):
                for k in range(K):
                    dist = np.sum((xi - protos[c, k])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c
            if best_c == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc:
            best_acc, best_K = acc, K

    protos = np.zeros((C, best_K, d), dtype=np.float32)
    for c in range(C):
        Xc = Xz[y == c]
        if len(Xc) == 0:
            protos[c] = 0
        elif len(Xc) <= best_K:
            protos[c, :len(Xc)] = Xc
            if len(Xc) < best_K:
                protos[c, len(Xc):] = Xc[rng.integers(len(Xc), size=best_K - len(Xc))]
        else:
            protos[c] = kmeans(Xc, best_K, rng)

    magic = 0x544F5250  # 'PROT'
    out = struct.pack("<4i", magic, d, best_K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K < 1 || K > 20) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)C * K * d);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *protos = sd + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mu[j]) / sd[j];
        float best_dist = 1e30f;
        unsigned char best_c = 0;
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *p = protos + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xz[j] - p[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    return 0;
}
```