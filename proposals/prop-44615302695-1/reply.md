```hypothesis
Family: label-lvq (Learning Vector Quantization). Two prototypes per class (20 total) trained with LVQ1 on z-scored 8x8 pixels. LVQ adapts prototypes to class boundaries, improving over linear's global hyperplanes, while inference stays fast (20×64 multiply-adds). Fixed seed, 30 epochs, decaying LR. Failure: dev accuracy ≤ 96.1% (label-linear) or Ir/item ≥ 6000.
```
```python
import json, struct, sys
import numpy as np

PROTOS_PER_CLASS = 2
LR_INIT = 0.1
EPOCHS = 30
LR_DECAY = 0.95
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_bytes)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    # Z-score per feature
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_z = ((X - mean) / std).astype(np.float32)

    # Initialize prototypes: 2 per class
    rng = np.random.default_rng(SEED)
    protos = []
    proto_labels = []
    for c in range(C):
        class_mask = (y == c)
        if not class_mask.any():
            continue
        class_mean = X_z[class_mask].mean(axis=0)
        for _ in range(PROTOS_PER_CLASS):
            proto = class_mean + rng.normal(0, 0.01, d).astype(np.float32)
            protos.append(proto)
            proto_labels.append(c)
    protos = np.array(protos, dtype=np.float32)
    proto_labels = np.array(proto_labels, dtype=np.uint8)
    n_proto = len(protos)

    # LVQ1 training
    lr = LR_INIT
    indices = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(indices)
        for idx in indices:
            x = X_z[idx]
            y_true = y[idx]
            dists = np.sum((protos - x) ** 2, axis=1)
            nearest = np.argmin(dists)
            if proto_labels[nearest] == y_true:
                protos[nearest] += lr * (x - protos[nearest])
            else:
                protos[nearest] -= lr * (x - protos[nearest])
        lr *= LR_DECAY

    # Save weights (little-endian)
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, n_proto, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_labels.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_proto = hd[2], C = hd[3];
    if (d != 64 || n_proto <= 0 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 8L * d + 4L * n_proto * d + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std = mean + d;
    float *protos = std + d;
    unsigned char *proto_labels = (unsigned char *)(protos + (long)n_proto * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mean[j]) / std[j];
        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < n_proto; p++) {
            float dist = 0.0f;
            const float *proto = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xz[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_label = proto_labels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```