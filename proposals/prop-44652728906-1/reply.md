```hypothesis
Prototype-based classifier (label-proto) with per-class k-means centroids. Optdigits digits form tight clusters; a few prototypes per class capture intra-class variation (writing styles) better than a linear boundary. With P=3..5 prototypes/class, expect >0.97 dev accuracy at lower CPU cost than label-linear (distance to 30-50 prototypes vs 10 dot-products). Failure: dev accuracy ≤0.9613 or instructions/item ≥5058.
```

```python
"""Prototype family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses prototypes-per-class P on tuning split (last 20%),
then trains on all data. Exports: magic 0x50524F54, d, P, C; float32 prototypes[C][P][d], p_norms[C][P]."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_ITER = 100
TOL = 1e-4
CANDIDATE_P = [1, 2, 3, 4, 5, 6, 7, 8]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(Xc, P, rng):
    """Xc: (n_c, d) float32. Return centroids (P, d) float32."""
    n_c, d = Xc.shape
    if n_c <= P:
        # duplicate points if needed
        centroids = Xc[:P]
        if n_c < P:
            centroids = np.vstack([centroids, rng.uniform(0, 16, (P - n_c, d)).astype(np.float32)])
        return centroids
    # k-means++ initialization
    idx = rng.integers(n_c)
    cents = [Xc[idx]]
    for _ in range(1, P):
        dist2 = np.min(np.sum((Xc[:, None] - np.array(cents))**2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n_c, p=probs)
        cents.append(Xc[idx])
    cents = np.array(cents, dtype=np.float32)
    for _ in range(MAX_ITER):
        # assign
        diff = Xc[:, None] - cents[None]  # (n_c, P, d)
        dist2 = np.sum(diff**2, axis=2)
        labels = np.argmin(dist2, axis=1)
        # update
        new_cents = np.zeros_like(cents)
        counts = np.zeros(P, dtype=np.int32)
        for i in range(n_c):
            c = labels[i]
            new_cents[c] += Xc[i]
            counts[c] += 1
        for k in range(P):
            if counts[k] > 0:
                new_cents[k] /= counts[k]
            else:
                new_cents[k] = cents[k]  # keep old
        if np.max(np.sqrt(np.sum((new_cents - cents)**2, axis=1))) < TOL:
            cents = new_cents
            break
        cents = new_cents
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    Xf = X_train.astype(np.float32)
    Xf_tune = X_tune.astype(np.float32)

    best_P = 1
    best_acc = -1.0
    for P in CANDIDATE_P:
        # train prototypes on train split
        protos = np.zeros((C, P, d), dtype=np.float32)
        for c in range(C):
            Xc = Xf[y_train == c]
            protos[c] = kmeans_class(Xc, P, rng)
        # precompute norms
        p_norms = np.sum(protos**2, axis=2)  # (C, P)
        # evaluate on tune
        correct = 0
        for i in range(len(X_tune)):
            x = Xf_tune[i]
            x_norm = np.dot(x, x)
            best_dist = 1e30
            best_c = 0
            for c in range(C):
                for p in range(P):
                    dist = x_norm + p_norms[c, p] - 2.0 * np.dot(x, protos[c, p])
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c
            if best_c == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc:
            best_acc = acc
            best_P = P

    # retrain on all data with best_P
    Xf_all = X.astype(np.float32)
    protos = np.zeros((C, best_P, d), dtype=np.float32)
    for c in range(C):
        Xc = Xf_all[y == c]
        protos[c] = kmeans_class(Xc, best_P, rng)
    p_norms = np.sum(protos**2, axis=2)

    # write weights
    magic = 0x50524F54  # 'PROT'
    header = struct.pack("<4i", magic, d, best_P, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(protos.astype("<f4").tobytes())
        f.write(p_norms.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Prototype family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class of nearest prototype.
 * Weights format: uint32 magic 0x50524F54, int32 d, int32 P, int32 C;
 * float32 prototypes[C][P][d]; float32 p_norms[C][P]. */
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (C < 1 || C > 256 || P < 1 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C * P * d + (long)C * P);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *protos = (const float *)(hd + 4);
    const float *p_norms = protos + (long)C * P * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float x_norm = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = (float)xi[j];
            x_norm += v * v;
        }
        float best_dist = 1e30f;
        unsigned char best_c = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float dot = 0.0f;
                for (int j = 0; j < d; j++) dot += (float)xi[j] * proto[j];
                float dist = x_norm + p_norms[c * P + p] - 2.0f * dot;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    return 0;
}
```