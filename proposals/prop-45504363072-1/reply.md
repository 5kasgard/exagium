```hypothesis
label-lvq: Learning Vector Quantization with 3 prototypes per class (30 total). Prototypes trained via mini-batch Adam on softmax(-distance) loss. Inference: exact nearest prototype (30×64 FMAs). Captures non-linear digit manifolds at ~3× linear cost; targets ≥0.975 dev accuracy. Fails if prototypes collapse to class means or optimization diverges.
```
```python
import json, math, struct, sys

import numpy as np

PROTOS_PER_CLASS = 3
SEED, LR, L2, BATCH, EPOCHS = 1, 1e-2, 1e-4, 64, 200


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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize prototypes: 3 per class around class mean
    P = PROTOS_PER_CLASS * C
    protos = np.zeros((P, d), dtype=np.float32)
    proto_class = np.zeros(P, dtype=np.uint8)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            mean = X[mask].mean(axis=0)
            for k in range(PROTOS_PER_CLASS):
                idx = c * PROTOS_PER_CLASS + k
                protos[idx] = mean + rng.normal(0, 0.1, d).astype(np.float32)
                proto_class[idx] = c
        else:
            for k in range(PROTOS_PER_CLASS):
                idx = c * PROTOS_PER_CLASS + k
                protos[idx] = rng.uniform(0, 16, d).astype(np.float32)
                proto_class[idx] = c

    # Adam state
    m = np.zeros_like(protos)
    v = np.zeros_like(protos)
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # Distances: (B, P) = sum((xb[:,None,:] - protos[None,:,:])^2, axis=2)
            diff = xb[:, None, :] - protos[None, :, :]           # (B, P, d)
            dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)  # (B, P)

            # Logits = -dist2
            logits = -dist2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (B, P)

            # Gradient wrt protos: dL/dprotos = 2 * sum_b (probs[b] - onehot) * (protos - xb)
            # Vectorized:
            # target probs: for each sample, target class prototypes get -1, others 0? No.
            # Loss = -log sum_{p in class_y} exp(-||x-proto_p||^2) + log sum_all exp(-||x-proto_p||^2)
            # This is softmax over prototypes with class pooling. Simpler: treat each prototype as a class,
            # but we want class-level supervision. Use softmax over prototypes, then group by class?
            # Standard LVQ: move closest correct closer, closest incorrect away.
            # Here we do gradient on prototype softmax cross-entropy with true class as target distribution
            # over prototypes: uniform over correct-class prototypes.
            target = np.zeros((len(yb), P), dtype=np.float32)
            for i, cls in enumerate(yb):
                target[i, proto_class == cls] = 1.0 / PROTOS_PER_CLASS

            grad_probs = (probs - target) / len(yb)  # (B, P)
            # dL/dprotos_p = sum_b grad_probs[b,p] * d(-dist2)/dprotos_p
            # d(-dist2)/dprotos_p = 2*(xb - protos_p)
            grad = 2.0 * np.einsum('bp,bpd->pd', grad_probs, diff)  # (P, d)
            grad += L2 * protos

            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            protos -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, P, C; protos[P][d] float32 LE; proto_class[P] uint8
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, P, C)
    out += protos.astype("<f4").tobytes()
    out += proto_class.astype("<u1").tobytes()
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d != 64 || C != 10 || P <= 0 || P > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)P * d * 4 + (long)P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *protos = (const float *)(wb + 16);
    const unsigned char *proto_class = wb + 16 + (long)P * d * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int p = 0; p < P; p++) {
            const float *pr = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_class = proto_class[p];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```