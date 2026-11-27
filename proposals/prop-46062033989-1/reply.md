```hypothesis
Family: label-proto (prototype classifier with k-means++ initialization and LVQ fine-tuning).
Why: Optdigits has distinct writing styles per digit; a few prototypes per class capture non-linear boundaries better than a single linear separator, while inference (distance to ~30 prototypes) is much faster than k-NN on all training data. Tuning k (prototypes per class) on the 20% holdout selects capacity.
Failure signal: Dev accuracy ≤ label-linear (0.9613) or inference cost ≥ label-mlp16 (8685 Ir/item).
```
```python
"""label-proto: prototype classifier (k-means++ init + LVQ1 fine-tune) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Per-pixel z-scoring on training split. Tuning: last 20% of train.in chooses k in {1,2,3,4,5} prototypes/class.
Final model trained on all data with best k. Weights: magic, d, P, C, mean[d], std[d], proto[P][d], pcls[P]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K_CHOICES = [1, 2, 3, 4, 5]
LVQ_EPOCHS = 10
LVQ_LR0 = 0.3


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_pp(X, k, rng):
    """X: (n, d), return (k, d) centroids. Deterministic given rng state."""
    n, d = X.shape
    cents = np.empty((k, d), np.float32)
    cents[0] = X[rng.integers(n)]
    for i in range(1, k):
        dist2 = np.min(np.sum((X[:, None, :] - cents[:i][None, :, :]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        cents[i] = X[rng.choice(n, p=probs)]
    # Lloyd iterations
    for _ in range(20):
        dist2 = np.sum((X[:, None, :] - cents[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, np.int32)
        for j in range(n):
            c = assign[j]
            new_cents[c] += X[j]
            counts[c] += 1
        for c in range(k):
            if counts[c]:
                new_cents[c] /= counts[c]
            else:
                new_cents[c] = X[rng.integers(n)]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents.astype(np.float32)


def lvq_fit(protos, pcls, Z, y, rng, epochs, lr0):
    """protos: (P, d), pcls: (P,), Z: (n, d), y: (n,)"""
    P, d = protos.shape
    for ep in range(epochs):
        lr = lr0 * (1.0 - ep / epochs)
        order = rng.permutation(len(Z))
        for idx in order:
            z, yy = Z[idx], y[idx]
            dist2 = np.sum((protos - z) ** 2, axis=1)
            best = np.argmin(dist2)
            if pcls[best] == yy:
                protos[best] += lr * (z - protos[best])
            else:
                protos[best] -= lr * (z - protos[best])
    return protos


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning (file order)
    split = int(n * 0.8)
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    # Preprocessing stats on train split
    mu = X_tr.mean(axis=0).astype(np.float32)
    sd = X_tr.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    def normalize(Xx):
        return ((Xx.astype(np.float32) - mu) / sd).astype(np.float32)

    Z_tr = normalize(X_tr)
    Z_tune = normalize(X_tune)

    best_k, best_acc = 1, -1.0
    for k in K_CHOICES:
        protos_list, pcls_list = [], []
        for c in range(C):
            Zc = Z_tr[y_tr == c]
            if len(Zc) == 0:
                continue
            cents = kmeans_pp(Zc, min(k, len(Zc)), rng)
            protos_list.append(cents)
            pcls_list.append(np.full(len(cents), c, np.uint8))
        protos = np.vstack(protos_list)
        pcls = np.concatenate(pcls_list)

        # LVQ fine-tune on train split
        protos = lvq_fit(protos, pcls, Z_tr, y_tr, rng, LVQ_EPOCHS, LVQ_LR0)

        # Evaluate on tune
        dist2 = np.sum((Z_tune[:, None, :] - protos[None, :, :]) ** 2, axis=2)
        pred = pcls[np.argmin(dist2, axis=1)]
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_k = acc, k

    # Retrain on full data with best_k
    mu_full = X.mean(axis=0).astype(np.float32)
    sd_full = X.std(axis=0).astype(np.float32)
    sd_full[sd_full == 0] = 1.0
    Z_full = ((X.astype(np.float32) - mu_full) / sd_full).astype(np.float32)

    protos_list, pcls_list = [], []
    for c in range(C):
        Zc = Z_full[y == c]
        if len(Zc) == 0:
            continue
        cents = kmeans_pp(Zc, min(best_k, len(Zc)), rng)
        protos_list.append(cents)
        pcls_list.append(np.full(len(cents), c, np.uint8))
    protos = np.vstack(protos_list)
    pcls = np.concatenate(pcls_list)
    protos = lvq_fit(protos, pcls, Z_full, y, rng, LVQ_EPOCHS, LVQ_LR0)

    P = protos.shape[0]
    magic = 0x50524F54  # 'PROT'
    # Write weights: magic, d, P, C, mean[d], std[d], protos[P][d], pcls[P]
    out = struct.pack("<4i", magic, d, P, C)
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += pcls.astype("<u1").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixel counts). Output record: 1 byte class.
 * Weights format: magic(0x50524F54), d, P, C, mean[d], std[d], proto[P][d], pcls[P] (all little-endian). */
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16 + (size_t)d * 4 * 2 + (size_t)P * d * 4 + (size_t)P;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std = mean + d;
    const float *proto = (const float *)(std + d);
    const uint8_t *pcls = (const uint8_t *)(proto + (size_t)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *zbuf = malloc((size_t)d * sizeof(float));
    float *dist = malloc((size_t)P * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) zbuf[j] = (xi[j] - mean[j]) / std[j];
        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            const float *pr = proto + (size_t)p * d;
            for (int j = 0; j < d; j++) {
                float diff = zbuf[j] - pr[j];
                sum += diff * diff;
            }
            dist[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < P; p++) if (dist[p] < dist[best]) best = p;
        unsigned char out = pcls[best];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```