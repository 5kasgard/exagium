```hypothesis
Family: label-lvq (Learning Vector Quantization). Optdigits forms tight per-digit clusters in pixel space; LVQ learns a small set of prototypes per class (5×10=50) that model non-linear boundaries better than a linear map, while inference is just 50×64 squared-distance calculations. If dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 10000, the approach fails.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Model: per-feature z-score + K prototypes per class (tuned on last 20% of train).
Exports: magic 0x4C565131, n_proto, d, C; float32 mean[d], std[d]; float32 proto[n_proto][d]; uint8 proto_class[n_proto]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_START, LR_END, EPOCHS = 0.1, 0.001, 30
CANDIDATES = [3, 5, 7, 10]  # prototypes per class


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def lvq_train(Z, y, n_proto, C, rng):
    d = Z.shape[1]
    # init: pick first n_proto samples per class after shuffle
    proto = np.zeros((n_proto * C, d), np.float32)
    proto_cls = np.zeros(n_proto * C, np.uint8)
    for c in range(C):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        take = idx[:n_proto]
        if len(take) < n_proto:
            take = np.resize(take, n_proto)
        proto[c * n_proto:(c + 1) * n_proto] = Z[take]
        proto_cls[c * n_proto:(c + 1) * n_proto] = c
    # LVQ1
    n = Z.shape[0]
    for ep in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (ep / max(1, EPOCHS - 1))
        order = rng.permutation(n)
        for i in order:
            z = Z[i]
            dists = np.sum((proto - z) ** 2, axis=1)
            p = int(np.argmin(dists))
            if proto_cls[p] == y[i]:
                proto[p] += lr * (z - proto[p])
            else:
                proto[p] -= lr * (z - proto[p])
    return proto, proto_cls


def lvq_acc(Z, y, proto, proto_cls):
    dists = ((Z[:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
    pred = proto_cls[np.argmin(dists, axis=1)]
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # split: last 20% for tuning (file order)
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    mu, sd = zscore_fit(X_tr)
    Z_tr = zscore_apply(X_tr.astype(np.float32), mu, sd)
    Z_va = zscore_apply(X_va.astype(np.float32), mu, sd)

    rng = np.random.default_rng(SEED)
    best_k, best_acc, best_proto, best_cls = CANDIDATES[0], -1.0, None, None
    for k in CANDIDATES:
        proto, pcls = lvq_train(Z_tr, y_tr, k, C, rng)
        acc = lvq_acc(Z_va, y_va, proto, pcls)
        if acc > best_acc:
            best_acc, best_k, best_proto, best_cls = acc, k, proto, pcls

    # retrain on full train with best_k
    mu_full, sd_full = zscore_fit(X)
    Z_full = zscore_apply(X.astype(np.float32), mu_full, sd_full)
    rng2 = np.random.default_rng(SEED)
    proto, pcls = lvq_train(Z_full, y_full, best_k, C, rng2)

    n_proto = best_k * C
    magic = 0x4C565131
    out = struct.pack("<4i", magic, n_proto, d, C)
    out += mu_full.tobytes() + sd_full.tobytes()
    out += best_proto.astype("<f4").tobytes()
    out += best_cls.astype(np.uint8).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Model: magic(0x4C565131), n_proto, d, C; mean[d], std[d] (float32);
 *        proto[n_proto][d] (float32); proto_class[n_proto] (uint8). */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_proto = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    int32_t d = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    int32_t C = wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24;
    if (n_proto <= 0 || d != 64 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * n_proto * d + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std = mean + d;
    const float *proto = (const float *)(wb + 16 + 4L * d * 2);
    const unsigned char *pcls = (const unsigned char *)(proto + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *znorm = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * n_proto);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) znorm[j] = (xi[j] - mean[j]) / std[j];
        for (int p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *w = proto + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = znorm[j] - w[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < n_proto; p++) if (dists[p] < dists[best]) best = p;
        unsigned char out = pcls[best];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```