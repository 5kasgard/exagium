```hypothesis
Family: label-quad (factorized quadratic model). It captures pairwise pixel interactions via a low-rank symmetric quadratic form on z-scored pixels, adding only O(d*R) parameters (R=4) over a linear model. Should improve accuracy on optdigits where stroke intersections matter, while keeping inference fast (table lookups + R inner products). If dev accuracy ≤ label-linear (0.9613) or instructions/item exceed label-mlp16 (~8700), the family does not justify its complexity.
```

```python
"""label-quad: factorized quadratic model for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal bytes to z-scores, learns linear + rank-R quadratic form.
Exports per-byte lookup tables: magic 0x51554144, d, R, C; float32 L[d][256][C], U[d][256][R], W2[R][C], bias[C]."""
import json, math, struct, sys

import numpy as np

R = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100
MAGIC = 0x51554144  # "QUAD" in little-endian


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Build z-score encoding for all 256 byte values per pixel
    E = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        E[j] = (np.arange(256, dtype=np.float32) - mu) / sd

    # Encoded training matrix
    Z = E[np.arange(d), X.T].T.astype(np.float32)  # (n, d)

    # Parameters
    W1 = (rng.normal(0, 0.01, (d, C))).astype(np.float32)
    A  = (rng.normal(0, 0.01, (d, R))).astype(np.float32)
    W2 = (rng.normal(0, 0.01, (R, C))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    params = [W1, A, W2, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Z.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            linear = z @ W1 + bias
            za = z @ A
            quad = za * za
            scores = linear + quad @ W2
            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = probs / len(yy)
            # gradients
            grad_bias = grad_scores.sum(0)
            grad_W1 = z.T @ grad_scores + L2 * W1
            grad_quad = grad_scores @ W2.T
            grad_za = 2.0 * za * grad_quad
            grad_A = z.T @ grad_za + L2 * A
            grad_W2 = za.T @ grad_quad + L2 * W2
            grads = [grad_W1, grad_A, grad_W2, grad_bias]
            # Adam update
            t = 1  # simplified: constant step count per batch; deterministic enough
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fold encoding into per-byte tables
    L = (E[:, :, np.newaxis] * W1[np.newaxis, :, :]).astype(np.float32)   # (d, 256, C)
    U = (E[:, :, np.newaxis] * A[np.newaxis, :, :]).astype(np.float32)    # (d, 256, R)

    # Write weights.bin
    total_floats = d * 256 * (C + R) + R * C + C
    expected_bytes = 16 + 4 * total_floats
    out = struct.pack("<4i", MAGIC, d, R, C)
    out += L.tobytes()
    out += U.tobytes()
    out += W2.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    assert len(out) == expected_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-quad inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights format: magic 0x51554144, int32 d, R, C; float32 L[d][256][C], U[d][256][R], W2[R][C], bias[C]. */
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
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected_floats = (long)d * 256 * (C + R) + (long)R * C + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *U = L + (long)d * 256 * C;
    const float *W2 = U + (long)d * 256 * R;
    const float *bias = W2 + (long)R * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sumU = malloc(sizeof(float) * R);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, bias, C * sizeof(float));
        memset(sumU, 0, R * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *Lj = L + ((long)j * 256 + xi[j]) * C;
            const float *Uj = U + ((long)j * 256 + xi[j]) * R;
            for (int c = 0; c < C; c++) acc[c] += Lj[c];
            for (int r = 0; r < R; r++) sumU[r] += Uj[r];
        }
        for (int r = 0; r < R; r++) {
            float quad = sumU[r] * sumU[r];
            const float *W2r = W2 + (long)r * C;
            for (int c = 0; c < C; c++) acc[c] += quad * W2r[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```