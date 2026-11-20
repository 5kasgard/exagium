```hypothesis
Family: label-proto (per-class k-means prototypes with 1-NN). Why: optdigits has compact within-class clusters; 5 prototypes/class (50 total) capture non-linear boundaries better than linear models while keeping inference at ~3200 ops/item (vs 2.2M for full k-NN). Fixed seed k-means per class is deterministic and fast to train. Fail condition: dev accuracy < 0.961 (label-linear) or inference > 10k Ir/item.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 5
MAX_ITER = 100
TOL = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(Xc, k, rng):
    n = Xc.shape[0]
    if n <= k:
        cents = Xc.astype(np.float32)
        if n < k:
            extra = rng.uniform(0, 16, (k - n, Xc.shape[1])).astype(np.float32)
            cents = np.vstack([cents, extra])
        return cents
    idx = rng.choice(n, k, replace=False)
    cents = Xc[idx].astype(np.float32).copy()
    for _ in range(MAX_ITER):
        diff = Xc[:, None, :] - cents[None, :, :]
        dist2 = (diff * diff).sum(axis=2)
        assign = dist2.argmin(axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, np.int32)
        for i in range(n):
            c = assign[i]
            new_cents[c] += Xc[i]
            counts[c] += 1
        for j in range(k):
            if counts[j] > 0:
                new_cents[j] /= counts[j]
            else:
                new_cents[j] = rng.uniform(0, 16, Xc.shape[1])
        shift = np.sqrt(((new_cents - cents) ** 2).sum(axis=1)).max()
        cents = new_cents
        if shift < TOL:
            break
    return cents.astype(np.float32)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d).astype(np.float32)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_M, best_acc = PROTOS_PER_CLASS, -1.0
    for M in [3, 5, 7, 10]:
        protos = []
        for c in range(C):
            Xc = X_tr[y_tr == c]
            cents = kmeans_class(Xc, M, rng)
            protos.append(cents)
        protos = np.stack(protos)  # [C, M, d]
        correct = 0
        for i in range(len(X_val)):
            xi = X_val[i]
            diff = protos - xi[None, None, :]
            dist2 = (diff * diff).sum(axis=2)
            c_pred = dist2.argmin() // M
            if c_pred == y_val[i]:
                correct += 1
        acc = correct / len(X_val)
        if acc > best_acc:
            best_acc, best_M = acc, M
    protos_final = []
    for c in range(C):
        Xc = X[y == c]
        cents = kmeans_class(Xc, best_M, rng)
        protos_final.append(cents)
    protos_final = np.stack(protos_final).astype("<f4")
    magic = 0x544F5250  # 'PROT'
    hdr = struct.pack("<4i", magic, d, C, best_M)
    open(sys.argv[4], "wb").write(hdr + protos_final.tobytes())

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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], M = hd[3];
    if (d != 64 || C != 10 || M < 1 || M > 32) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * C * M * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f; int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                const float *p = protos + ((long)c * M + m) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = (float)xi[j] - p[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_class = c; }
            }
        }
        unsigned char out = (unsigned char)best_class;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```