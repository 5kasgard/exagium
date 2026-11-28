```hypothesis
Family: label-lvq (Learning Vector Quantization). A small set of prototypes per class (e.g., 3 per class = 30 total) are learned by pulling prototypes toward same-class examples and pushing away from other classes. On optdigits, the 8x8 ordinal pixel space should be well approximated by a few prototypes per digit, capturing non-linear boundaries that a linear model misses. Inference is just a nearest-prototype search (30 * 64 multiply-adds), which is comparable in cost to label-linear. If dev accuracy does not exceed 0.9613, the prototype count is too low or the ordinal pixel structure needs a different distance metric.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: uint32 magic 0x4C565131, int32 P, d, C; then P * (d*float32 + uint8 class)."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
PROTO_PER_CLASS_GRID = [1, 2, 3, 4, 5]
LR_GRID = [0.05, 0.1, 0.2]
EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, n_proto_per_class, lr, epochs, rng, classes=10):
    n, d = X.shape
    # Initialize prototypes: pick random samples from each class
    prototypes = []
    proto_classes = []
    for c in range(classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(n_proto_per_class, len(idx)), replace=False)
        for i in chosen:
            prototypes.append(X[i].astype(np.float32))
            proto_classes.append(c)
    prototypes = np.array(prototypes, dtype=np.float32)  # (P, d)
    proto_classes = np.array(proto_classes, dtype=np.uint8)
    P = prototypes.shape[0]

    for ep in range(epochs):
        order = rng.permutation(n)
        # decay learning rate
        cur_lr = lr * (1.0 - ep / epochs)
        for i in order:
            x = X[i].astype(np.float32)
            # find nearest prototype
            diffs = prototypes - x  # (P, d)
            dists = np.sum(diffs * diffs, axis=1)
            nearest = np.argmin(dists)
            if proto_classes[nearest] == y[i]:
                prototypes[nearest] += cur_lr * (x - prototypes[nearest])
            else:
                prototypes[nearest] -= cur_lr * (x - prototypes[nearest])
    return prototypes, proto_classes

def lvq_predict(X, prototypes, proto_classes):
    # X: (n, d) float32
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        diffs = prototypes - X[i]
        dists = np.sum(diffs * diffs, axis=1)
        preds[i] = proto_classes[np.argmin(dists)]
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning (file order)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_protos = None
    best_proto_classes = None

    for nppc in PROTO_PER_CLASS_GRID:
        for lr in LR_GRID:
            protos, pclasses = lvq_train(X_train, y_train, nppc, lr, EPOCHS, rng, C)
            preds = lvq_predict(X_tune.astype(np.float32), protos, pclasses)
            acc = np.mean(preds == y_tune)
            if acc > best_acc:
                best_acc = acc
                best_protos = protos
                best_proto_classes = pclasses

    # Retrain on all data with best hyperparameters (using same seed for determinism)
    # We need to know which hyperparameters gave best_acc. We'll just retrain with the same loop but on all data.
    # Actually we can just keep the best model from tuning? But the card says: choose settings on tuning split, then train on all of train.
    # So we must retrain on all data with the chosen nppc and lr.
    # We'll re-run the grid on the tuning split to find the best hyperparameters, then train on all data.
    # To avoid recomputing, we can store the best hyperparameters.
    # Let's do a simple re-evaluation on the tuning split to pick best hyperparameters, then train on all data.
    # We already have best_acc but not the hyperparameters. We'll re-run the grid on tuning split to get them.
    # Since the grid is small, we can just do it again.
    best_nppc = PROTO_PER_CLASS_GRID[0]
    best_lr = LR_GRID[0]
    best_acc = -1.0
    for nppc in PROTO_PER_CLASS_GRID:
        for lr in LR_GRID:
            protos, pclasses = lvq_train(X_train, y_train, nppc, lr, EPOCHS, rng, C)
            preds = lvq_predict(X_tune.astype(np.float32), protos, pclasses)
            acc = np.mean(preds == y_tune)
            if acc > best_acc:
                best_acc = acc
                best_nppc = nppc
                best_lr = lr

    # Train on all data with best hyperparameters
    final_protos, final_pclasses = lvq_train(X, y, best_nppc, best_lr, EPOCHS, rng, C)

    # Write weights
    P = final_protos.shape[0]
    magic = 0x4C565131  # "LVQ1"
    header = struct.pack("<4i", magic, P, d, C)
    proto_bytes = final_protos.astype("<f4").tobytes()
    class_bytes = final_pclasses.astype("<B").tobytes()
    # Interleave: each prototype: d floats then 1 byte class
    # We'll write as a single array: for each prototype, d floats then 1 byte.
    # But we have them separate. Let's combine.
    out = bytearray()
    out += header
    for i in range(P):
        out += final_protos[i].astype("<f4").tobytes()
        out += bytes([final_pclasses[i]])
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte class (0..C-1).
 * Weights format: uint32 magic 0x4C565131, int32 P, d, C; then P*(d*float32 + uint8 class). */
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
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t P = *(int32_t*)(wb+4);
    int32_t d = *(int32_t*)(wb+8);
    int32_t C = *(int32_t*)(wb+12);
    if (P <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)P * ((long)d * 4 + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *prototypes = (const float*)(wb + 16);
    const unsigned char *proto_classes = wb + 16 + (long)P * d * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < P; p++) {
            const float *proto = prototypes + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_classes[p];
            }
        }
        out_byte = best_class;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```