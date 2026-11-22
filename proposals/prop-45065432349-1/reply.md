```hypothesis
Family: label-proto (prototypical classifier with per-class k-means).
Why: Optdigits has distinct visual modes per digit; k-means prototypes in z-scored pixel space capture non-linear boundaries better than a linear model while keeping inference fast (O(C*K*d) with K=3). The 30 centroids (3 per class) should exceed label-linear's 96.13% accuracy at similar or lower CPU cost.
Failure: If dev accuracy ≤ 0.9613 or instructions/item ≥ label-linear's 5058, the family does not improve on the baseline.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
K = 3
MAX_ITER = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    if n <= K:
        cents = X[:K].copy()
        if n < K:
            cents[n:] = X[rng.integers(n, size=K - n)]
        return cents
    idx = rng.choice(n, size=K, replace=False)
    cents = X[idx].copy()
    for _ in range(max_iter):
        diff = X[:, None, :] - cents[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_c = np.zeros_like(cents)
        cnt = np.zeros(K, dtype=np.int32)
        for i in range(n):
            k = assign[i]
            new_c[k] += X[i]
            cnt[k] += 1
        for k in range(K):
            if cnt[k] == 0:
                new_c[k] = X[rng.integers(n)]
                cnt[k] = 1
            else:
                new_c[k] /= cnt[k]
        if np.allclose(cents, new_c):
            break
        cents = new_c
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd

    Z = ((X - mu) * inv_sd).astype(np.float32)

    all_cents = []
    for c in range(C):
        Zc = Z[y == c]
        rng = np.random.default_rng(SEED + c + 1)
        cents = kmeans(Zc, K, rng, MAX_ITER)
        all_cents.append(cents)
    cents = np.vstack(all_cents).astype(np.float32)
    cent_norms = np.sum(cents * cents, axis=1).astype(np.float32)

    magic = 0x50524F54
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr
    out += mu.astype(np.float32).tobytes()
    out += inv_sd.astype(np.float32).tobytes()
    out += cents.tobytes()
    out += cent_norms.tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)C * K * (d + 1));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *inv_sd = mu + d;
    const float *cents = inv_sd + d;
    const float *cent_norms = cents + (long)C * K * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) * inv_sd[j];
        float z2 = 0.0f;
        for (int j = 0; j < d; j++) z2 += z[j] * z[j];
        float best_dist = 1e30f;
        int best_idx = 0;
        for (int ck = 0; ck < C * K; ck++) {
            const float *c = cents + (long)ck * d;
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += z[j] * c[j];
            float dist = z2 - 2.0f * dot + cent_norms[ck];
            if (dist < best_dist) { best_dist = dist; best_idx = ck; }
        }
        unsigned char cls = best_idx / K;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```