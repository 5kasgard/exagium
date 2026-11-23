```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of prototypes per class (tuned 1–5/class on the 20% tuning split), capturing multi-modal digit shapes better than a linear model while keeping inference extremely cheap (≈30 prototypes × 64 dims). On optdigits, prototypes naturally represent distinct writing styles. Expected to beat label-linear (0.9613) and approach label-mlp16 (0.9770) at lower CPU cost. Fails if dev accuracy ≤ 0.9613 or instructions/item exceed label-mlp16.
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

def zscore(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def lvq_train(X, y, n_proto_per_class, lr_init=0.1, epochs=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    classes = np.unique(y)
    prototypes = []
    proto_labels = []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=n_proto_per_class, replace=False)
        prototypes.append(X[chosen])
        proto_labels.append(np.full(n_proto_per_class, c))
    prototypes = np.vstack(prototypes)
    proto_labels = np.concatenate(proto_labels)
    n_proto = len(prototypes)

    lr = lr_init
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in order:
            x = X[i]
            yy = y[i]
            dists = np.sum((prototypes - x) ** 2, axis=1)
            best = np.argmin(dists)
            if proto_labels[best] == yy:
                prototypes[best] += lr * (x - prototypes[best])
            else:
                prototypes[best] -= lr * (x - prototypes[best])
        lr *= 0.95
    return prototypes, proto_labels

def evaluate(X, y, prototypes, proto_labels):
    correct = 0
    for i in range(len(X)):
        dists = np.sum((prototypes - X[i]) ** 2, axis=1)
        best = np.argmin(dists)
        if proto_labels[best] == y[i]:
            correct += 1
    return correct / len(X)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = len(X)
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    X_train_norm, mean, std = zscore(X_train.astype(np.float32))
    X_tune_norm = zscore(X_tune.astype(np.float32), mean, std)[0]

    best_acc = -1
    best_nppc = 3
    rng = np.random.default_rng(SEED)
    for nppc in [1, 2, 3, 4, 5]:
        protos, labels = lvq_train(X_train_norm, y_train, nppc, rng=rng)
        acc = evaluate(X_tune_norm, y_tune, protos, labels)
        if acc > best_acc:
            best_acc = acc
            best_nppc = nppc

    X_all_norm, mean_all, std_all = zscore(X.astype(np.float32))
    rng = np.random.default_rng(SEED)
    final_protos, final_labels = lvq_train(X_all_norm, y, best_nppc, rng=rng)

    magic = 0x4C565131
    n_proto = len(final_protos)
    out = struct.pack("<4i", magic, d, n_proto, C)
    out += mean_all.astype("<f4").tobytes()
    out += std_all.astype("<f4").tobytes()
    out += final_protos.astype("<f4").tobytes()
    out += final_labels.astype(np.uint8).tobytes()
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    long expected = 16 + 8L * d + 4L * n_proto * d + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *protos = (const float *)(std + d);
    const unsigned char *labels = (const unsigned char *)(protos + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * n_proto);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *proto = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xnorm[j] - proto[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }
        int best = 0;
        for (int p = 1; p < n_proto; p++) {
            if (dists[p] < dists[best]) best = p;
        }
        unsigned char out = labels[best];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```