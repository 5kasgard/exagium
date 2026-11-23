```hypothesis
Family: label-rff (Random Fourier Features). It approximates an RBF kernel by projecting z-scored pixels
onto 16 random frequencies (32 features: cos/sin), then learns a linear layer on top. This adds
non-linearity to capture pixel correlations while keeping inference a few thousand FLOPs.
If dev accuracy ≤ 0.9613 (label-linear) the hypothesis fails.
```

```python
"""label-rff: Random Fourier Features for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal bytes by z-scoring.
Projects onto D=16 random frequencies (W[d, D], b[D]) fixed at seed 123.
RFF features: cos(Z@W+b), sin(Z@W+b) -> 2D features.
Trains logistic regression on 2D features with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports: magic 0x52464631, d, D, C; float32 T[d][256][D] (z-score * W), b[D], V[2D][C], c[C]."""
import json, math, struct, sys

import numpy as np

D = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
PROJ_SEED = 123


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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)
    proj_rng = np.random.default_rng(PROJ_SEED)

    # per-byte encoding: ordinal -> z-score
    means = np.zeros(d, np.float64)
    stds = np.ones(d, np.float64)
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            means[j], stds[j] = mu, sd
        else:
            raise ValueError("only ordinal supported")
    # random projection matrix
    W = (proj_rng.normal(0, 1, (d, D)) / np.sqrt(d)).astype(np.float32)
    b = (proj_rng.uniform(0, 2 * math.pi, D)).astype(np.float32)

    # build lookup table T[j, v, k] = ((v - mean_j)/std_j) * W[j, k]
    T = np.zeros((d, 256, D), np.float32)
    for j in range(d):
        z = (np.arange(256, dtype=np.float32) - means[j]) / stds[j]
        T[j] = z[:, None] * W[j]  # (256, D)

    # compute RFF features for training
    Z = (X.astype(np.float32) - means.astype(np.float32)) / stds.astype(np.float32)  # (n, d)
    proj = Z @ W + b  # (n, D)
    rff = np.concatenate([np.cos(proj), np.sin(proj)], axis=1).astype(np.float32)  # (n, 2D)
    n, k = rff.shape

    # linear layer on RFF features
    V = np.zeros((k, C), np.float32)
    c = np.zeros(C, np.float32)
    params = [V, c]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            zz, yy = rff[idx], y[idx]
            g = softmax_grad(zz @ V + c, yy)
            grads = [zz.T @ g + L2 * V, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # export
    magic, d_out, D_out, C_out = 0x52464631, d, D, C
    out = struct.pack("<4i", magic, d_out, D_out, C_out)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += c.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format: magic 0x52464631, int32 d, D, C; float32 T[d][256][D], b[D], V[2D][C], c[C].
 * Computes proj_k = b_k + sum_j T[j, x_j, k]; rff = cos(proj), sin(proj); scores = c + rff @ V. */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * D
        + 4L * D
        + 4L * (2 * D) * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * D;
    const float *V = b + D;
    const float *c = V + (long)2 * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *proj = malloc(sizeof(float) * D);
    float *rff = malloc(sizeof(float) * 2 * D);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < D; u++) proj[u] = b[u];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) proj[u] += tj[u];
        }
        for (int u = 0; u < D; u++) {
            float p = proj[u];
            rff[2*u] = cosf(p);
            rff[2*u+1] = sinf(p);
        }
        for (int cidx = 0; cidx < C; cidx++) sc[cidx] = c[cidx];
        for (int u = 0; u < 2*D; u++) {
            float h = rff[u];
            const float *vu = V + (long)u * C;
            for (int cidx = 0; cidx < C; cidx++) sc[cidx] += h * vu[cidx];
        }
        unsigned char best = 0;
        for (int cidx = 1; cidx < C; cidx++) if (sc[cidx] > sc[best]) best = (unsigned char)cidx;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```