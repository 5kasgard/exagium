```hypothesis
Family: label-lvq (Learning Vector Quantization with 3 prototypes per class).
Why: LVQ positions prototypes near decision boundaries, capturing non-linear structure in the 8×8 digit space that a linear model misses. With 30 prototypes total, it should exceed label-linear's 0.9613 dev accuracy while keeping inference cost low (~3× linear, ~15k Ir/item).
Failure: If dev accuracy ≤ 0.9613 or inference cost > 50k Ir/item, the approach does not meet the goal.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
N_PROTOS_PER_CLASS = 3
EPOCHS = 80
LR_INIT = 0.05
WINDOW = 0.3

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
    n, d = X_bytes.shape
    C = card["classes"]

    X = X_bytes.astype(np.float32) / 16.0
    rng = np.random.default_rng(SEED)

    protos = []
    proto_classes = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            for _ in range(N_PROTOS_PER_CLASS):
                protos.append(rng.uniform(0, 1, d).astype(np.float32))
                proto_classes.append(c)
        else:
            chosen = rng.choice(idx, size=min(N_PROTOS_PER_CLASS, len(idx)), replace=False)
            for i in chosen:
                protos.append(X[i].copy())
                proto_classes.append(c)
            while len([p for p, pc in zip(protos, proto_classes) if pc == c]) < N_PROTOS_PER_CLASS:
                base = protos[[i for i, pc in enumerate(proto_classes) if pc == c][0]]
                protos.append(base + rng.normal(0, 0.01, d).astype(np.float32))
                proto_classes.append(c)

    protos = np.array(protos, dtype=np.float32)
    proto_classes = np.array(proto_classes, dtype=np.int64)
    n_protos = protos.shape[0]

    proto_norms = np.sum(protos * protos, axis=1)

    for epoch in range(EPOCHS):
        lr = LR_INIT * (1.0 - epoch / EPOCHS)
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            true_c = y[i]
            dots = protos @ x
            dists = proto_norms - 2.0 * dots
            nearest_two = np.argpartition(dists, 2)[:2]
            nearest_two = nearest_two[np.argsort(dists[nearest_two])]
            c1, c2 = proto_classes[nearest_two[0]], proto_classes[nearest_two[1]]
            d1, d2 = dists[nearest_two[0]], dists[nearest_two[1]]

            if c1 != c2 and d1 > 0 and d2 > 0:
                ratio = min(d1, d2) / max(d1, d2)
                if ratio > (1.0 - WINDOW) / (1.0 + WINDOW):
                    if c1 == true_c and c2 != true_c:
                        protos[nearest_two[0]] += lr * (x - protos[nearest_two[0]])
                        protos[nearest_two[1]] -= lr * (x - protos[nearest_two[1]])
                        proto_norms[nearest_two[0]] = np.sum(protos[nearest_two[0]] ** 2)
                        proto_norms[nearest_two[1]] = np.sum(protos[nearest_two[1]] ** 2)
                    elif c2 == true_c and c1 != true_c:
                        protos[nearest_two[1]] += lr * (x - protos[nearest_two[1]])
                        protos[nearest_two[0]] -= lr * (x - protos[nearest_two[0]])
                        proto_norms[nearest_two[1]] = np.sum(protos[nearest_two[1]] ** 2)
                        proto_norms[nearest_two[0]] = np.sum(protos[nearest_two[0]] ** 2)

    magic = 0x4C565131
    out = struct.pack("<4i", magic, n_protos, d, C)
    out += protos.astype("<f4").tobytes()
    out += proto_classes.astype(np.uint8).tobytes()
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_protos = hd[1], d = hd[2], C = hd[3];
    if (n_protos <= 0 || d != 64 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)n_protos * d * 4 + n_protos;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *protos = (const float *)(hd + 4);
    const unsigned char *proto_classes = (const unsigned char *)((const char *)protos + (long)n_protos * d * 4);

    float *proto_norms = malloc(sizeof(float) * n_protos);
    for (int i = 0; i < n_protos; i++) {
        float sum = 0.0f;
        const float *p = protos + (long)i * d;
        for (int j = 0; j < d; j++) sum += p[j] * p[j];
        proto_norms[i] = sum;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *input_vec = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) input_vec[j] = xi[j] / 16.0f;

        float x_norm = 0.0f;
        for (int j = 0; j < d; j++) x_norm += input_vec[j] * input_vec[j];

        float best_dist = 1e30f;
        int best_proto = 0;
        for (int pi = 0; pi < n_protos; pi++) {
            const float *p = protos + (long)pi * d;
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += input_vec[j] * p[j];
            float dist = proto_norms[pi] - 2.0f * dot;
            if (dist < best_dist) {
                best_dist = dist;
                best_proto = pi;
            }
        }
        unsigned char best_class = proto_classes[best_proto];
        put_rec(stdout, &best_class, 1);
    }

    free(proto_norms);
    free(input_vec);
    return 0;
}
```