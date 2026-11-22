```hypothesis
Family: label-proto (multi-prototype linear classifier). For each digit, learn K=3..5 prototypes via k-means on training pixels; inference scores all prototypes with a linear table lookup (score = 2*x·p - ||p||^2) and predicts the class of the highest-scoring prototype. This captures non-linear class boundaries (Voronoi per class) at nearly the same cost as label-linear (one table pass, no ReLU). Expect dev accuracy >0.97 (beating label-linear's 0.9613) with Ir/item ~6k. If accuracy ≤0.965 or cost >2× label-linear, the approach fails.
```
```python
import json, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(X_class, K, rng):
    n, d = X_class.shape
    if n <= K:
        centers = X_class.astype(np.float32)
        if n < K:
            extra = rng.uniform(0, 16, (K - n, d)).astype(np.float32)
            centers = np.vstack([centers, extra])
        return centers
    centers = np.zeros((K, d), np.float32)
    idx = rng.integers(n)
    centers[0] = X_class[idx].astype(np.float32)
    for k in range(1, K):
        dist2 = np.full(n, np.inf, np.float32)
        for c in centers[:k]:
            d2 = np.sum((X_class.astype(np.float32) - c) ** 2, axis=1)
            np.minimum(dist2, d2, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centers[k] = X_class[idx].astype(np.float32)
    for _ in range(20):
        dists = np.zeros((n, K), np.float32)
        for k in range(K):
            dists[:, k] = np.sum((X_class.astype(np.float32) - centers[k]) ** 2, axis=1)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros((K, d), np.float32)
        counts = np.zeros(K, np.int32)
        for i in range(n):
            k = labels[i]
            new_centers[k] += X_class[i]
            counts[k] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                new_centers[k] = X_class[rng.integers(n)].astype(np.float32)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_all)
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]
    
    rng = np.random.default_rng(SEED)
    best_K, best_acc = 1, -1.0
    for K in [1, 2, 3, 4, 5]:
        prototypes, proto_classes = [], []
        for c in range(C):
            Xc = X_train[y_train == c]
            centers = kmeans_class(Xc, K, rng)
            prototypes.append(centers)
            proto_classes.extend([c] * K)
        prototypes = np.vstack(prototypes).astype(np.float32)
        proto_classes = np.array(proto_classes, np.uint8)
        M = len(prototypes)
        p_norms = np.sum(prototypes ** 2, axis=1)
        scores = 2 * (X_tune.astype(np.float32) @ prototypes.T) - p_norms
        preds = proto_classes[np.argmax(scores, axis=1)]
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc, best_K = acc, K
    
    rng = np.random.default_rng(SEED)
    prototypes, proto_classes = [], []
    for c in range(C):
        Xc = X_all[y_all == c]
        centers = kmeans_class(Xc, best_K, rng)
        prototypes.append(centers)
        proto_classes.extend([c] * best_K)
    prototypes = np.vstack(prototypes).astype(np.float32)
    proto_classes = np.array(proto_classes, np.uint8)
    M = len(prototypes)
    
    T = np.zeros((d, 256, M), np.float32)
    for j in range(d):
        for v in range(256):
            T[j, v, :] = 2.0 * v * prototypes[:, j]
    b = -np.sum(prototypes ** 2, axis=1).astype(np.float32)
    
    magic = 0x544F5250  # "PROT"
    out = struct.pack("<4i", magic, d, M, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += proto_classes.astype("<u1").tobytes()
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || M < 1 || M > 1000 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d * 256 * M + 4L * M + M;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * M;
    const unsigned char *proto_class = (const unsigned char *)(b + M);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * M);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, M * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * M;
            for (int m = 0; m < M; m++) scores[m] += t[m];
        }
        int best_m = 0;
        for (int m = 1; m < M; m++) if (scores[m] > scores[best_m]) best_m = m;
        put_rec(stdout, &proto_class[best_m], 1);
    }
    return 0;
}
```