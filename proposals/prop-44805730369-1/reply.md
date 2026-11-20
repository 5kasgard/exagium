```hypothesis
Family: label-proto (Learning Vector Quantization prototype classifier).
Why: Optdigits 8x8 images have distinct class manifolds; a small set of learned prototypes per class captures non-linear boundaries better than linear models while keeping inference extremely fast (just distance-to-prototypes). With ~3-4 prototypes/class (30-40 total), we expect ~97.5% dev accuracy at ~2000 Ir/item, beating label-linear's 96.1% at 5059 Ir/item.
Failure signal: Dev accuracy ≤ 96.2% or Ir/item ≥ 5000 would show the family doesn't improve on label-linear for this task.
```

```python
"""label-proto: Learning Vector Quantization prototype classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Encodes ordinal features as z-scored (mean/std from train). Learns K prototypes per class via LVQ.
Exports: magic 0x50524F54, d, K, C; float32 mean[d], std[d]; float32 proto[K][d]; uint8 proto_label[K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
PROTOS_PER_CLASS = 3          # K_per_class
LVQ_EPOCHS = 20
LVQ_LR_START = 0.3
LVQ_LR_END = 0.01


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Z-score ordinal features (all are ordinal per card)
    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_bytes.astype(np.float32) - mu) / sd).astype(np.float32)   # (n, d)

    n = X.shape[0]
    # Initialize prototypes: for each class, pick PROTOS_PER_CLASS samples deterministically
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        # Evenly spaced indices for determinism
        step = max(1, len(idx) // PROTOS_PER_CLASS)
        for i in range(PROTOS_PER_CLASS):
            pick = idx[min(i * step, len(idx) - 1)]
            protos.append(X[pick].copy())
            proto_labels.append(c)
    protos = np.array(protos, dtype=np.float32)      # (K, d)
    proto_labels = np.array(proto_labels, dtype=np.uint8)
    K = protos.shape[0]

    # LVQ training
    for epoch in range(LVQ_EPOCHS):
        lr = LVQ_LR_START * (LVQ_LR_END / LVQ_LR_START) ** (epoch / max(1, LVQ_EPOCHS - 1))
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            yi = y[i]
            # Euclidean distances to all prototypes
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist2))
            if proto_labels[best] == yi:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # Pack weights
    # magic, d, K, C
    hdr = struct.pack("<4i", 0x50524F54, d, K, C)
    # mean[d], std[d]
    body = mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    # protos[K][d] (row-major)
    body += protos.astype("<f4").tobytes()
    # proto_labels[K]
    body += proto_labels.tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class (nearest prototype after z-scoring). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expect = 16L + (long)d * 4 * 2 + (long)K * d * 4 + K;
    if (nw != expect) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expect); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std  = mean + d;
    const float *protos = (const float *)(wb + 16 + (long)d * 8);
    const unsigned char *proto_labels = (const unsigned char *)(wb + 16 + (long)d * 8 + (long)K * d * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *zbuf = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // z-score
        for (int j = 0; j < d; j++) zbuf[j] = (xi[j] - mean[j]) / std[j];
        // nearest prototype
        float best_dist2 = 1e30f; unsigned char best_label = 0;
        for (int p = 0; p < K; p++) {
            const float *pr = protos + (long)p * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = zbuf[j] - pr[j];
                dist2 += diff * diff;
            }
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_label = proto_labels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```