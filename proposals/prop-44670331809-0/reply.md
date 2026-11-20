```hypothesis
Propose label-fm1: a rank-1 factorization machine added to a linear model.
On this tabular task with 14 binned features, pairwise interactions are known to help (GBDT wins).
A rank-1 FM adds only d*256 + 2*C parameters and ~2x the inference ops of linear.
If dev accuracy does not exceed label-linear by ≥0.5% (i.e., <0.8574), the interaction term is insufficient.
```

```python
"""Factorization Machine rank 1 + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D3131, d, k=1, C; then float32 arrays:
  W_lin[d][256][C], V[d][256][k], W_int[k][C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 1
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters
    W_lin = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    W_int = (rng.normal(0, 0.01, (K, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W_lin, V, W_int, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            # Linear part
            lin = b + np.sum(W_lin[np.arange(d)[:, None], xb.T, :], axis=0)  # (bs, C)

            # FM part (rank 1)
            v = V[np.arange(d)[:, None], xb.T, 0]  # (d, bs)
            s_f = v.sum(axis=0)                     # (bs,)
            sq_f = (v * v).sum(axis=0)              # (bs,)
            inter = 0.5 * (s_f * s_f - sq_f)        # (bs,)
            lin += inter[:, None] * W_int[0]        # (bs, C)

            # Softmax + loss gradient
            scores = lin - lin.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            g = probs / bs  # (bs, C)

            # Gradients
            grad_b = g.sum(axis=0)

            grad_W_lin = np.zeros_like(W_lin)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_W_lin[j], vals, g)

            grad_W_int = np.zeros_like(W_int)
            grad_W_int[0] = (inter[:, None] * g).sum(axis=0)

            grad_V = np.zeros_like(V)
            # d(inter)/dV[j,v] = s_f - V[j,v]
            for j in range(d):
                vals = xb[:, j]
                # For each sample, factor = sum_c g * W_int[0,c] = g @ W_int[0]
                factor = g @ W_int[0]  # (bs,)
                # grad_V[j, v] += sum_{i: xb[i,j]=v} factor[i] * (s_f[i] - V[j,v])
                # We can accumulate per value
                for v_val in range(256):
                    mask = (vals == v_val)
                    if mask.any():
                        grad_V[j, v_val, 0] = (factor[mask] * (s_f[mask] - V[j, v_val, 0])).sum()

            grads = [grad_W_lin, grad_V, grad_W_int, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_adam = state[i]
                m *= 0.9; m += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # L2 regularization (weight decay)
                if i != 3:  # no decay on bias
                    p -= LR * L2 * p

    # Export
    magic = 0x464D3131
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W_int.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine rank 1 + linear for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (K != 1 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *V = W_lin + (long)d * 256 * C;
    const float *W_int = V + (long)d * 256 * K;
    const float *b = W_int + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_f = malloc(sizeof(float) * K);
    float *sum_sq_f = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < K; f++) { sum_f[f] = 0.0f; sum_sq_f[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *wlin = W_lin + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += wlin[c];

            for (int f = 0; f < K; f++) {
                float val = V[(long)j * 256 * K + v * K + f];
                sum_f[f] += val;
                sum_sq_f[f] += val * val;
            }
        }

        for (int f = 0; f < K; f++) {
            float inter = 0.5f * (sum_f[f] * sum_f[f] - sum_sq_f[f]);
            for (int c = 0; c < C; c++) scores[c] += inter * W_int[f * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_f); free(sum_sq_f);
    if (x) free(x);
    return 0;
}
```