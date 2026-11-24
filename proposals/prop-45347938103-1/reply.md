```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means prototypes).
Why: Optdigits 8x8 pixels have spatial structure; prototypes capture non-linear class boundaries better than linear models.
Cost scales as O(C*k*d) ~ 10*3*64 = 1920 distances/item, between label-linear and label-mlp16.
If dev accuracy ≤ 0.965 or Ir/item > 12000, the hypothesis fails.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
K_CANDIDATES = [1, 2, 3, 4, 5]
KMEANS_ITERS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_pp(X, k, rng):
    n, d = X.shape
    cents = np.zeros((k, d), dtype=X.dtype)
    idx = rng.integers(n)
    cents[0] = X[idx]
    for i in range(1, k):
        diff = X[:, None, :] - cents[:i]
        dist2 = np.einsum('ijk,ijk->ij', diff, diff).min(axis=1)
        if dist2.sum() == 0:
            idx = rng.integers(n)
        else:
            probs = dist2 / dist2.sum()
            idx = rng.choice(n, p=probs)
        cents[i] = X[idx]
    return cents

def kmeans(X, k, rng, max_iter=30):
    cents = kmeans_pp(X, k, rng)
    for _ in range(max_iter):
        diff = X[:, None, :] - cents
        dist2 = np.einsum('ijk,ijk->ij', diff, diff)
        labels = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_cents[i] = X[mask].mean(axis=0)
            else:
                new_cents[i] = X[rng.integers(X.shape[0])]
        if np.allclose(cents, new_cents):
            cents = new_cents
            break
        cents = new_cents
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C, n = card["input"]["bytes"], card["classes"], X.shape[0]
    n_tune = n // 5
    X_tr, y_tr = X[:n - n_tune], y[:n - n_tune]
    X_te, y_te = X[n - n_tune:], y[n - n_tune:]
    rng = np.random.default_rng(SEED)
    best_k, best_acc = 1, -1.0
    for k in K_CANDIDATES:
        mu = X_tr.mean(axis=0).astype(np.float32)
        sd = X_tr.std(axis=0).astype(np.float32)
        sd[sd == 0] = 1.0
        Z_tr = (X_tr.astype(np.float32) - mu) / sd
        Z_te = (X_te.astype(np.float32) - mu) / sd
        protos = np.zeros((C, k, d), dtype=np.float32)
        for c in range(C):
            Xc = Z_tr[y_tr == c]
            if len(Xc) == 0:
                continue
            if len(Xc) < k:
                protos[c] = np.tile(Xc.mean(axis=0), (k, 1)) + rng.normal(0, 0.01, (k, d)).astype(np.float32)
            else:
                protos[c] = kmeans(Xc, k, rng, KMEANS_ITERS)
        correct = 0
        for i in range(len(Z_te)):
            dist2 = ((protos - Z_te[i]) ** 2).sum(axis=2)
            if dist2.min(axis=1).argmin() == y_te[i]:
                correct += 1
        acc = correct / len(Z_te)
        if acc > best_acc:
            best_acc, best_k = acc, k
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = (X.astype(np.float32) - mu) / sd
    protos = np.zeros((C, best_k, d), dtype=np.float32)
    for c in range(C):
        Xc = Z[y == c]
        if len(Xc) == 0:
            continue
        if len(Xc) < best_k:
            protos[c] = np.tile(Xc.mean(axis=0), (best_k, 1)) + rng.normal(0, 0.01, (best_k, d)).astype(np.float32)
        else:
            protos[c] = kmeans(Xc, best_k, rng, KMEANS_ITERS)
    out = struct.pack("<4i", 0x544F5250, d, best_k, C)
    out += mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes() + protos.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx=0, N=atol(argv[3]), pos=0;
    int32_t *hd = (int32_t*)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d=hd[1], k=hd[2], C=hd[3];
    if (d<=0 || k<=0 || C<=0 || C>256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L*d*2 + 4L*(long)C*k*d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mu = (const float*)(hd+4);
    const float *sd = mu + d;
    const float *proto = sd + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float)*d);
    float *dist = malloc(sizeof(float)*C);
    for (long i=0; i<N; i++) {
        uint32_t rlen; const unsigned char *xi = next_rec(x, nx, &pos, &rlen);
        if ((int)rlen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rlen, d); return 3; }
        for (int j=0; j<d; j++) z[j] = (xi[j] - mu[j]) / sd[j];
        for (int c=0; c<C; c++) {
            float best = 1e30f;
            for (int p=0; p<k; p++) {
                const float *pr = proto + ((long)c*k + p)*d;
                float sum = 0.0f;
                for (int j=0; j<d; j++) { float diff = z[j] - pr[j]; sum += diff*diff; }
                if (sum < best) best = sum;
            }
            dist[c] = best;
        }
        unsigned char best = 0;
        for (int c=1; c<C; c++) if (dist[c] < dist[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```