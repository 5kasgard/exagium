```hypothesis
Family: label-proto (per-class k-means prototypes). Optdigits digits form tight clusters; 3 prototypes/class (30 total) capture non-linear boundaries better than linear while being 100x smaller than full 1-NN. Inference is pure integer-indexed float32 distance loops (no libm, no exp, no cos). Fails if dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 10000.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
K_CANDIDATES = [1, 2, 3, 4, 5]
MAX_ITER = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, seed, max_iter=MAX_ITER):
    """Deterministic k-means++ + Lloyd on float32 X (n, d). Returns centroids (K, d) float32."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Xf = X.astype(np.float32)
    cents = np.zeros((K, d), np.float32)
    # k-means++ init
    idx = rng.integers(n)
    cents[0] = Xf[idx]
    for k in range(1, K):
        diff = Xf[:, None, :] - cents[:k][None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest = np.min(dist2, axis=1)
        probs = nearest / nearest.sum()
        cdf = np.cumsum(probs)
        u = rng.random()
        idx = int(np.searchsorted(cdf, u))
        cents[k] = Xf[idx]
    # Lloyd
    for _ in range(max_iter):
        diff = Xf[:, None, :] - cents[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, np.int32)
        for i in range(n):
            c = labels[i]
            new_cents[c] += Xf[i]
            counts[c] += 1
        for c in range(K):
            if counts[c]:
                new_cents[c] /= counts[c]
            else:
                idx = rng.integers(n)
                new_cents[c] = Xf[idx]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents


def accuracy(X, y, cents, cent_labels):
    n = X.shape[0]
    Xf = X.astype(np.float32)
    correct = 0
    for i in range(n):
        diff = Xf[i] - cents
        dist2 = np.sum(diff * diff, axis=1)
        pred = cent_labels[np.argmin(dist2)]
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split = last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_K, best_acc = 1, -1.0
    for K in K_CANDIDATES:
        cents_list, labs_list = [], []
        for c in range(C):
            Xc = X_tr[y_tr == c]
            if len(Xc) == 0:
                continue
            cents = kmeans(Xc, K, SEED + c * 1000)
            cents_list.append(cents)
            labs_list.append(np.full(K, c, dtype=np.uint8))
        if not cents_list:
            continue
        cents_all = np.vstack(cents_list)
        labs_all = np.concatenate(labs_list)
        acc = accuracy(X_tune, y_tune, cents_all, labs_all)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on all data with best_K
    cents_list, labs_list = [], []
    for c in range(C):
        Xc = X[y == c]
        cents = kmeans(Xc, best_K, SEED + c * 1000)
        cents_list.append(cents)
        labs_list.append(np.full(best_K, c, dtype=np.uint8))
    cents_all = np.vstack(cents_list).astype(np.float32)
    labs_all = np.concatenate(labs_list).astype(np.uint8)
    n_proto = cents_all.shape[0]

    # weights: magic, d, C, K, n_proto; then prototypes (n_proto*d float32), labels (n_proto uint8)
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<5i", magic, d, C, best_K, n_proto)
    out += cents_all.astype("<f4").tobytes()
    out += labs_all.astype("<u1").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3], n_proto = hd[4];
    if (d != 64 || C != 10 || K < 1 || K > 5 || n_proto != C * K) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    long expected = 20L + (long)n_proto * d * 4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *cents = (const float *)(wb + 20);
    const unsigned char *labs = wb + 20 + (long)n_proto * d * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f; unsigned char best_lab = 0;
        for (int p = 0; p < n_proto; p++) {
            const float *cp = cents + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float dval = (float)xi[j] - cp[j];
                dist += dval * dval;
            }
            if (dist < best_dist) { best_dist = dist; best_lab = labs[p]; }
        }
        put_rec(stdout, &best_lab, 1);
    }
    return 0;
}
```