```hypothesis
Family: label-lvq (Learning Vector Quantization with 5 prototypes/class).
Why: LVQ excels on small image datasets like optdigits; 50 prototypes × 64 dims = 3200 FLOPs/item (faster than label-mlp16), and prototype adaptation sharpens decision boundaries beyond linear.
Failure: Dev accuracy ≤ 0.9613 (label-linear) or cost ≥ label-mlp16.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. 5 prototypes per class, LVQ1 with lr decay."""
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 5
LR_START, LR_END, EPOCHS = 0.1, 0.01, 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"]).astype(np.float32)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Initialize prototypes: stratified random sample per class
    protos, proto_labels = [], []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(PROTOS_PER_CLASS, len(idx)), replace=False)
        protos.append(X[chosen])
        proto_labels.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)
    proto_labels = np.concatenate(proto_labels)
    n_proto = len(protos)

    # LVQ1 training
    order = np.arange(n)
    for epoch in range(EPOCHS):
        lr = LR_START * (LR_END / LR_START) ** (epoch / max(1, EPOCHS - 1))
        rng.shuffle(order)
        for i in order:
            x = X[i]
            yi = y[i]
            # squared Euclidean distances
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            if proto_labels[best] == yi:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])

    # Export: magic, n_proto, d, C; prototypes (n_proto*d float32); labels (n_proto uint8)
    magic = 0x4C565131  # "LVQ1"
    hdr = struct.pack("<4i", magic, n_proto, d, C)
    out = hdr + protos.astype("<f4").tobytes() + proto_labels.astype(np.uint8).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (nearest prototype). */
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_proto = hd[1], d = hd[2], C = hd[3];
    if (n_proto <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + (long)n_proto * d * 4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    const float *protos = (const float *)(hd + 4);
    const unsigned char *plabels = (const unsigned char *)(protos + (long)n_proto * d);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist2 = 1e30f; unsigned char best_label = 0;
        for (int p = 0; p < n_proto; p++) {
            const float *w = protos + (long)p * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float delta = xi[j] - w[j];
                dist2 += delta * delta;
            }
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_label = plabels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```