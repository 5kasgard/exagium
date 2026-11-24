```hypothesis
Family: label-proto (prototype-based classifier). Learns K prototypes per class via k-means on the training data; inference is 1-NN among all prototypes. Captures multi-modal class distributions with O(K*C*d) operations per item. Should beat linear (0.9613) on optdigits because digits have multiple writing styles per class, while staying cheaper than label-mlp16 (8685 Ir/item) by avoiding hidden-layer matmul. Failure if dev accuracy ≤ 0.9613 or Ir/item ≥ 8685.
```

```python
"""label-proto: prototype classifier via per-class k-means.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x50524F54, int32 d, K, C; then C*K*d float32 prototypes (class-major)."""
import json, struct, sys
import numpy as np

SEED = 1
K_CANDIDATES = [1, 2, 4, 8, 16, 32]  # prototypes per class
MAX_ITER = 100
TOL = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng):
    """X: (n, d) float32, returns (K, d) float32 centroids."""
    n, d = X.shape
    if n <= K:
        # pad with zeros if not enough samples
        cents = np.zeros((K, d), dtype=np.float32)
        cents[:n] = X
        return cents
    # k-means++ initialization (deterministic via rng)
    cents = np.empty((K, d), dtype=np.float32)
    idx = rng.integers(n)
    cents[0] = X[idx]
    for k in range(1, K):
        dist2 = np.min(np.sum((X[:, None, :] - cents[:k])**2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        cents[k] = X[idx]
    # Lloyd's algorithm
    for _ in range(MAX_ITER):
        diff = X[:, None, :] - cents[None, :, :]  # (n, K, d)
        dist2 = np.sum(diff**2, axis=2)
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
                # reinitialize empty cluster randomly
                new_cents[k] = X[rng.integers(n)]
        shift = np.sqrt(np.sum((new_cents - cents)**2))
        cents = new_cents
        if shift < TOL:
            break
    return cents


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # tuning split: last 20%
    tune_size = n // 5
    tr_idx = slice(0, n - tune_size)
    tu_idx = slice(n - tune_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_tu, y_tu = X[tu_idx], y[tu_idx]

    rng_global = np.random.default_rng(SEED)
    best_K, best_acc = 1, -1.0
    for K in K_CANDIDATES:
        # train prototypes on training split
        protos = np.zeros((C, K, d), dtype=np.float32)
        for c in range(C):
            Xc = X_tr[y_tr == c].astype(np.float32)
            rng_c = np.random.default_rng(SEED + c + K * 1000)
            protos[c] = kmeans(Xc, K, rng_c)
        # evaluate on tuning split
        # flatten prototypes for distance computation
        flat_protos = protos.reshape(C * K, d)
        proto_classes = np.repeat(np.arange(C), K)
        # vectorized distance: (n_tu, C*K)
        diff = X_tu[:, None, :].astype(np.float32) - flat_protos[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        pred = proto_classes[np.argmin(dist2, axis=1)]
        acc = np.mean(pred == y_tu)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on full data with best_K
    protos = np.zeros((C, best_K, d), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c].astype(np.float32)
        rng_c = np.random.default_rng(SEED + c + best_K * 1000)
        protos[c] = kmeans(Xc, best_K, rng_c)

    # write weights.bin
    flat = protos.reshape(-1)  # C*K*d
    out = struct.pack("<4i", 0x50524F54, d, best_K, C)
    out += flat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8). Output record: 1 byte class.
 * Weights: magic 0x50524F54, int32 d, K, C; then C*K*d float32 prototypes (class-major). */
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)C * K * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *protos = (const float *)(hd + 4);  // [C*K][d]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_f = malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) in_f[j] = (float)xi[j];
        float best_dist = 1e30f;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                const float *proto = protos + ((long)c * K + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = in_f[j] - proto[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = c;
                }
            }
        }
        unsigned char out = (unsigned char)best_class;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```