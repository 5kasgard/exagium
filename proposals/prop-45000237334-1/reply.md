```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of discriminative prototypes per class in the z-scored pixel space (64D), giving non-linear decision boundaries with O(P*C*d) inference (P=3 prototypes/class → ~1920 FLOPs). It should beat label-linear (0.9613) on optdigits because handwritten digits have intra-class variation (writing styles) that multiple prototypes capture better than a single linear separator per class. Failure: dev accuracy ≤ 0.9613 or Ir/item > 2× label-linear.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encodes ordinal bytes as z-scored floats (like label-linear). Learns P prototypes per class
via LVQ1: for each sample, move nearest prototype toward/away based on class match.
Hyperparameter P (prototypes per class) chosen on last 20% of train (tuning split),
then retrained on all train. Exports: magic, d, P, C, mean[d], inv_std[d], proto[C][P][d]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LVQ_EPOCHS = 30
LVQ_LR0 = 0.05
LVQ_LR_DECAY = 0.95
P_CANDIDATES = [1, 2, 3, 4, 5]  # prototypes per class to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def zscore_encode(X, types):
    """Return (Z, mean, inv_std) where Z is float32 [n, d], mean/inv_std are float32 [d]."""
    n, d = X.shape
    Z = np.empty((n, d), dtype=np.float32)
    mean = np.empty(d, dtype=np.float32)
    inv_std = np.empty(d, dtype=np.float32)
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu = v.mean()
            sd = v.std()
            if sd == 0:
                sd = 1.0
            mean[j] = mu
            inv_std[j] = 1.0 / sd
            Z[:, j] = (v - mu) * inv_std[j]
        else:
            # categorical not expected here, but handle gracefully
            vals = np.unique(X[:, j])
            # one-hot would expand dims; fallback: ordinal coding
            mu = v.mean()
            sd = v.std() or 1.0
            mean[j] = mu
            inv_std[j] = 1.0 / sd
            Z[:, j] = (v - mu) * inv_std[j]
    return Z, mean, inv_std


def lvq_train(Z, y, C, P, rng, epochs=LVQ_EPOCHS, lr0=LVQ_LR0, decay=LVQ_LR_DECAY):
    """Train LVQ1. Returns prototypes [C, P, d] (float32)."""
    n, d = Z.shape
    # Initialize prototypes: for each class, pick P random samples from that class
    protos = np.zeros((C, P, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            # fallback: random noise
            protos[c] = rng.normal(0, 1, (P, d)).astype(np.float32)
        else:
            chosen = rng.choice(idx, size=P, replace=len(idx) < P)
            protos[c] = Z[chosen].astype(np.float32)
    lr = lr0
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in order:
            z = Z[i]
            yi = y[i]
            # find nearest prototype across all classes
            best_dist = float('inf')
            best_c, best_p = 0, 0
            for c in range(C):
                for p in range(P):
                    diff = protos[c, p] - z
                    dist = np.dot(diff, diff)
                    if dist < best_dist:
                        best_dist = dist
                        best_c, best_p = c, p
            # LVQ1 update
            sign = 1.0 if best_c == yi else -1.0
            protos[best_c, best_p] += sign * lr * (z - protos[best_c, best_p])
        lr *= decay
    return protos


def evaluate_protos(Z, y, protos):
    """Return accuracy of nearest-prototype classifier."""
    n = Z.shape[0]
    correct = 0
    for i in range(n):
        z = Z[i]
        best_dist = float('inf')
        best_c = 0
        C, P, _ = protos.shape
        for c in range(C):
            for p in range(P):
                diff = protos[c, p] - z
                dist = np.dot(diff, diff)
                if dist < best_dist:
                    best_dist = dist
                    best_c = c
        if best_c == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Encode all data once
    Z, mean, inv_std = zscore_encode(X, types)
    n = Z.shape[0]

    # Split for tuning: last 20% in file order
    split = int(n * 0.8)
    Z_train, y_train = Z[:split], y[:split]
    Z_tune, y_tune = Z[split:], y[split:]

    # Try each P on tuning split
    best_P = 1
    best_acc = -1.0
    for P in P_CANDIDATES:
        protos = lvq_train(Z_train, y_train, C, P, rng)
        acc = evaluate_protos(Z_tune, y_tune, protos)
        if acc > best_acc:
            best_acc = acc
            best_P = P

    # Retrain on all data with best P
    protos = lvq_train(Z, y, C, best_P, rng)

    # Pack weights: magic, d, P, C, mean[d], inv_std[d], protos[C][P][d] (all float32 LE)
    magic = 0x5051564C  # 'LVQP' little-endian
    out = struct.pack("<4I", magic, d, best_P, C)
    out += mean.astype("<f4").tobytes()
    out += inv_std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format (little-endian):
 *   uint32_t magic (0x5051564C), d, P, C
 *   float mean[d], inv_std[d]
 *   float proto[C][P][d]  (row-major: C blocks of P blocks of d floats)
 * For each item: z-score with mean/inv_std, find nearest prototype (squared L2), output its class. */
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
    if (nw < 16 || hd[0] != 0x5051564C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t d = hd[1], P = hd[2], C = hd[3];
    if (C < 1 || C > 256 || P == 0 || P > 256 || d == 0 || d > 1024) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + (size_t)d * 2 * 4 + (size_t)C * P * d * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mean = (const float *)(hd + 4);
    const float *inv_std = mean + d;
    const float *proto = inv_std + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = (float *)malloc(d * sizeof(float));
    if (!z) { fprintf(stderr, "oom\n"); exit(3); }
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        for (uint32_t j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) * inv_std[j];
        }
        float best_dist = INFINITY;
        unsigned char best_c = 0;
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t p = 0; p < P; p++) {
                const float *w = proto + ((size_t)c * P + p) * d;
                float dist = 0.0f;
                for (uint32_t j = 0; j < d; j++) {
                    float diff = w[j] - z[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    free(z);
    return 0;
}
```