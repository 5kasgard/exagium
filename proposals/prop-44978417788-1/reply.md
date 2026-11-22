```hypothesis
Family: label-quadratic. Adds per-feature quadratic terms (z^2) to the linear model's z-scored pixels, capturing non-monotonic intensity effects in handwritten digits (e.g., mid-range pixel values distinguish strokes from noise). Trains same Adam/L2 regimen as label-linear on 128-d expanded features (z, z^2 per pixel). Expects >96.13% dev accuracy (beating label-linear) at ~10k Ir/item (2x linear, near label-mlp16 speed). Fails if accuracy ≤96.13% or cost >15k Ir/item.
```
```python
"""label-quadratic: per-feature quadratic (z and z^2) + linear classifier for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Adam lr 3e-3, L2 1e-4, batch 64, epochs min(100, ceil(600000/n)).
Exports: magic 0x51554144, d, 0, C; float32 T_lin[d][256][C], T_quad[d][256][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            z_vals = (np.arange(256) - mu) / sd
            E = np.column_stack([z_vals, z_vals * z_vals])
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E_lin = np.zeros((256, len(vals)))
            E_lin[vals, np.arange(len(vals))] = 1.0
            E_quad = np.zeros((256, len(vals)))
            enc.append(np.column_stack([E_lin, E_quad]))

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    assert k == 2 * d

    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W1 + b1
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    T_lin = np.zeros((d, 256, C), np.float32)
    T_quad = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        w_lin = W1[2 * j]
        w_quad = W1[2 * j + 1]
        z_vals = enc[j][:, 0]
        z2_vals = enc[j][:, 1]
        T_lin[j] = np.outer(z_vals, w_lin).astype(np.float32)
        T_quad[j] = np.outer(z2_vals, w_quad).astype(np.float32)

    magic = 0x51554144
    out = struct.pack("<4i", magic, d, 0, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_quad.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-quadratic: per-feature quadratic (z and z^2) + linear classifier.
 * ./infer weights.bin in.rec N
 * Weights: magic 0x51554144, d, 0, C; float32 T_lin[d][256][C], T_quad[d][256][C], b[C].
 * Input: d bytes per record. Output: 1 byte class (argmax). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x51554144) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H != 0 || nw != 16 + 4L * ((long)d * 256 * C * 2 + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T_lin = (const float *)(hd + 4);
    const float *T_quad = T_lin + (long)d * 256 * C;
    const float *b = T_quad + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *tlin = T_lin + ((long)j * 256 + v) * C;
            const float *tquad = T_quad + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) {
                acc[c] += tlin[c] + tquad[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```