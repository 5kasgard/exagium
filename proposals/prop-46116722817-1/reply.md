```hypothesis
Family: label-lvq (Learning Vector Quantization). Prototype-based classifier with per-class prototypes; fast inference via Euclidean distance. Well-suited for optdigits because pixel patterns cluster naturally per digit. Fails if prototypes collapse or tuning split shows no gain over linear (dev accuracy ≤ 0.9613).
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
PROTO_GRID = [1, 2, 3, 4, 5]          # prototypes per class
LR_GRID = [0.1, 0.05, 0.01]
EPOCHS_GRID = [10, 20, 30]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, proto_per_class, lr, epochs, rng):
    n, d = X.shape
    C = 10
    P = proto_per_class * C
    # initialize prototypes: pick random samples per class
    protos = np.zeros((P, d), np.float32)
    proto_class = np.zeros(P, np.uint8)
    idx = 0
    for c in range(C):
        class_mask = (y == c)
        if not np.any(class_mask):
            continue
        choices = rng.choice(np.where(class_mask)[0], proto_per_class, replace=False)
        protos[idx:idx+proto_per_class] = X[choices]
        proto_class[idx:idx+proto_per_class] = c
        idx += proto_per_class
    # LVQ1
    for ep in range(epochs):
        alpha = lr * (1.0 - ep / epochs)
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            # find nearest prototype
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist2))
            if proto_class[best] == y[i]:
                protos[best] += alpha * (x - protos[best])
            else:
                protos[best] -= alpha * (x - protos[best])
    return protos, proto_class

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    # normalization on full train (but tuning split uses last 20% of file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    # compute mean/std on training portion
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    # normalize
    Xn = (X.astype(np.float32) - mu) / sd
    Xn_train = Xn[:split]
    Xn_tune = Xn[split:]

    best_acc = -1.0
    best_cfg = None
    best_protos = None
    best_pclass = None
    rng = np.random.default_rng(SEED)
    for i, (ppc, lr, ep) in enumerate([(p, l, e) for p in PROTO_GRID for l in LR_GRID for e in EPOCHS_GRID]):
        cfg_rng = np.random.default_rng(SEED + i + 1)
        protos, pclass = lvq_train(Xn_train, y_train, ppc, lr, ep, cfg_rng)
        # evaluate on tune
        diff = protos[:, None, :] - Xn_tune[None, :, :]  # P x Ntune x d
        dist2 = np.sum(diff * diff, axis=2)  # P x Ntune
        pred = pclass[np.argmin(dist2, axis=0)]
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_cfg = (ppc, lr, ep)
            best_protos = protos.copy()
            best_pclass = pclass.copy()

    # retrain on full data with best config
    cfg_rng = np.random.default_rng(SEED + 999)
    protos, pclass = lvq_train(Xn, y, best_cfg[0], best_cfg[1], best_cfg[2], cfg_rng)

    # save weights
    P = protos.shape[0]
    magic = 0x4C565131  # "LVQ1"
    hdr = struct.pack("<4i", magic, d, P, C)
    out = hdr + mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += pclass.astype(np.uint8).tobytes()
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t P = *(int32_t*)(wb + 8);
    int32_t C = *(int32_t*)(wb + 12);
    if (d != 64 || C != 10 || P <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * P * d + P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float*)(wb + 16);
    const float *sd = mu + d;
    const float *protos = sd + d;
    const unsigned char *pclass = (const unsigned char*)(protos + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xn = malloc(d * sizeof(float));
    float *dists = malloc(P * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xn[j] = (xi[j] - mu[j]) / sd[j];
        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            const float *pr = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xn[j] - pr[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < P; p++) if (dists[p] < dists[best]) best = p;
        unsigned char pred = pclass[best];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```