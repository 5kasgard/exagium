```hypothesis
Family: label-r1fm (Rank-1 Factorization Machine). A linear model per class augmented with a shared rank-1 quadratic interaction (zᵀu)² folded into per-byte lookup tables. Captures dominant pairwise interactions in Adult (e.g., education×occupation) at ~2× label-linear's memory and ~3× compute, targeting >0.855 dev accuracy. Failure: accuracy ≤0.853 shows rank-1 insufficient; would need rank-2.
```
```python
"""Rank-1 Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encodes ordinal as z-score, categorical as one-hot (unseen=0). Learns linear weights w_c and
quadratic weights u_c in encoded space, then exports per-byte lookup tables T_lin[c][j][v]
and T_quad[c][j][v] so inference is: score_c = sum_j T_lin[c][j][x_j] + (sum_j T_quad[c][j][x_j])^2.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x5231464D  # 'R1FM'


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
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Build encodings per feature
    enc = []
    dims = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        dims.append(E.shape[1])

    # Offsets in encoded space
    off = np.concatenate([[0], np.cumsum(dims)[:-1]])
    k = sum(dims)

    # Construct dense encoded matrix Z (n, k)
    n = X.shape[0]
    Z = np.empty((n, k), dtype=np.float32)
    for j in range(d):
        Z[:, off[j]:off[j] + dims[j]] = enc[j][X[:, j]]

    # Initialize parameters in encoded space: shape (k, C)
    w = np.zeros((k, C), dtype=np.float32)
    u = rng.normal(0, 0.01, (k, C)).astype(np.float32)

    # Adam state
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mu, vu = np.zeros_like(u), np.zeros_like(u)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            lin = z @ w  # (bsz, C)
            quad = z @ u  # (bsz, C)
            scores = lin + quad * quad  # (bsz, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(bsz), yy] -= 1.0
            grad_scores /= bsz  # (bsz, C)

            # Gradients
            grad_w = z.T @ grad_scores + L2 * w  # (k, C)
            grad_u = z.T @ (grad_scores * (2.0 * quad)) + L2 * u  # (k, C)

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            for param, grad, m, v in [(w, grad_w, mw, vw), (u, grad_u, mu, vu)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Fold into per-byte lookup tables
    T_lin = np.zeros((C, d, 256), dtype=np.float32)
    T_quad = np.zeros((C, d, 256), dtype=np.float32)
    for c in range(C):
        for j in range(d):
            w_slice = w[off[j]:off[j] + dims[j], c]
            u_slice = u[off[j]:off[j] + dims[j], c]
            T_lin[c, j] = enc[j] @ w_slice
            T_quad[c, j] = enc[j] @ u_slice

    # Pack weights: magic, d, C, reserved; then T_lin (C*d*256 floats), T_quad (C*d*256 floats)
    out = struct.pack("<4i", MAGIC, d, C, 0)
    out += T_lin.astype("<f4").tobytes()
    out += T_quad.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Rank-1 Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic(0x5231464D), d, C, reserved; float32 T_lin[C][d][256], T_quad[C][d][256].
 * Score_c = sum_j T_lin[c][j][x_j] + (sum_j T_quad[c][j][x_j])^2. */
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
    if (nw < 16 || hd[0] != 0x5231464D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 16 + 8L * C * d * 256) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T_lin = (const float *)(hd + 4);
    const float *T_quad = T_lin + (long)C * d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            float sum_lin = 0.0f, sum_quad = 0.0f;
            for (int j = 0; j < d; j++) {
                const float *tlin = T_lin + ((long)c * d + j) * 256;
                const float *tquad = T_quad + ((long)c * d + j) * 256;
                unsigned char v = xi[j];
                sum_lin += tlin[v];
                sum_quad += tquad[v];
            }
            scores[c] = sum_lin + sum_quad * sum_quad;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```