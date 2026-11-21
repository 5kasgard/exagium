```hypothesis
Propose label-proto: a prototype classifier with per-class k-means centroids.
Why: Optdigits pixels are ordinal; prototypes capture non-linear class structure better than linear decision boundaries.
If dev accuracy does not exceed label-linear (0.9613), the prototype approach is insufficient for this task.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
M_CANDIDATES = [3, 5, 7, 10]
KMEANS_ITERS = 20
MAGIC = 0x50524F54  # 'PROT'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, M, rng):
    n, d = X.shape
    if n <= M:
        return X.copy()
    idx = rng.choice(n, M, replace=False)
    cents = X[idx].copy()
    for _ in range(KMEANS_ITERS):
        dists = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(M, dtype=np.int32)
        for i in range(n):
            c = labels[i]
            new_cents[c] += X[i]
            counts[c] += 1
        for m in range(M):
            if counts[m] > 0:
                new_cents[m] /= counts[m]
            else:
                new_cents[m] = cents[m]
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

    # Z-score per feature (all ordinal)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    n = Z.shape[0]
    split = int(0.8 * n)
    Z_train, Z_tune = Z[:split], Z[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_M, best_acc = M_CANDIDATES[0], -1.0

    for M in M_CANDIDATES:
        protos = []
        for c in range(C):
            Xc = Z_train[y_train == c]
            cents = kmeans(Xc, M, rng)
            protos.append(cents)
        protos = np.vstack(protos)  # (C*M, d)
        proto_labels = np.repeat(np.arange(C), M)

        # Evaluate on tune
        diffs = Z_tune[:, None, :] - protos[None, :, :]
        dists = (diffs ** 2).sum(axis=2)
        preds = proto_labels[dists.argmin(axis=1)]
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc, best_M = acc, M

    # Retrain on all data with best_M
    protos = []
    for c in range(C):
        Xc = Z[y == c]
        cents = kmeans(Xc, best_M, rng)
        protos.append(cents)
    protos = np.vstack(protos).astype(np.float32)  # (C*best_M, d)

    # Write weights.bin
    # header: magic, d, M, C
    # then mean[d], std[d], protos[C*M][d]
    out = struct.pack("<4i", MAGIC, d, best_M, C)
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * (long)C * M * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *protos = sd + d;
    int P = C * M;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float zbuf[64];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) zbuf[j] = (xi[j] - mu[j]) / sd[j];
        float best_dist = 1e30f;
        unsigned char best_cls = 0;
        for (int p = 0; p < P; p++) {
            const float *pr = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = zbuf[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_cls = (unsigned char)(p / M);
            }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```