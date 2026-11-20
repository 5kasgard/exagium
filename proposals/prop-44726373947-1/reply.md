```hypothesis
Propose "label-proto": a multi-prototype nearest-neighbor classifier. For each class, learn P prototypes via k-means (P tuned on the last 20% of train). Inference computes squared Euclidean distance to all C×P prototypes and picks the class of the nearest. This adds non-linear boundaries while keeping inference extremely cheap (only 64×C×P subtractions/multiplies). Expected to surpass label-linear (0.9613) and approach label-mlp16 (0.9770) at a fraction of the cost. If dev accuracy does not exceed label-linear, the task is essentially linear or the prototype count is too small.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
MAX_ITER = 20
CANDIDATE_P = [1, 2, 3, 4, 5]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(data, P, rng):
    n = data.shape[0]
    if n <= P:
        protos = data[:P]
        if len(protos) < P:
            pad = np.zeros((P - len(protos), data.shape[1]), dtype=np.float32)
            protos = np.vstack([protos, pad])
        return protos
    idx = rng.choice(n, P, replace=False)
    protos = data[idx].copy()
    for _ in range(MAX_ITER):
        diff = data[:, None, :] - protos[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist, axis=1)
        new_protos = np.zeros_like(protos)
        counts = np.zeros(P, dtype=np.int32)
        for i in range(n):
            a = assign[i]
            new_protos[a] += data[i]
            counts[a] += 1
        for j in range(P):
            if counts[j] > 0:
                new_protos[j] /= counts[j]
            else:
                new_protos[j] = data[rng.integers(n)]
        if np.allclose(protos, new_protos):
            protos = new_protos
            break
        protos = new_protos
    return protos

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_all.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_tune, y_tune = X_all[train_size:], y_all[train_size:]

    best_P = 1
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    for P in CANDIDATE_P:
        all_protos = []
        for c in range(C):
            class_data = X_train[y_train == c].astype(np.float32)
            protos = kmeans_class(class_data, P, rng)
            all_protos.append(protos)
        all_protos = np.vstack(all_protos)
        correct = 0
        for i in range(len(X_tune)):
            x = X_tune[i].astype(np.float32)
            diff = all_protos - x
            dist = np.sum(diff * diff, axis=1)
            idx = np.argmin(dist)
            pred = idx // P
            if pred == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc:
            best_acc = acc
            best_P = P

    rng = np.random.default_rng(SEED)
    all_protos = []
    for c in range(C):
        class_data = X_all[y_all == c].astype(np.float32)
        protos = kmeans_class(class_data, best_P, rng)
        all_protos.append(protos)
    all_protos = np.vstack(all_protos)

    magic = 0x544F5250  # 'PROT'
    out = struct.pack("<4i", magic, d, best_P, C)
    out += all_protos.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (C < 1 || C > 256 || P < 1 || d <= 0 || nw != 16 + 4L * C * P * d) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *dist = malloc(sizeof(float) * C * P);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = (float)xi[j] - proto[j];
                    sum += diff * diff;
                }
                dist[c * P + p] = sum;
            }
        }
        unsigned char best_c = 0;
        float best_dist = dist[0];
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                float dval = dist[c * P + p];
                if (dval < best_dist) {
                    best_dist = dval;
                    best_c = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    return 0;
}
```