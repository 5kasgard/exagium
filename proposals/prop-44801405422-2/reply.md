```hypothesis
Factorization Machine (label-fm) captures pairwise position interactions via low-rank latent vectors,
critical for character n-gram statistics in text. The FM trick (sum-of-vectors squared) adds only O(K)
per-class work over linear inference. With K=8 it should beat label-linear (0.3375) on dev accuracy.
Failure would mean 16-byte context lacks pairwise signal or rank-8 is insufficient.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. FM with per-class latent vectors dim K=8.
Adam lr=0.01, L2=1e-4, batch=64, epochs=20.
Weights: int32 magic 0x464D4C31, d, C, K; float32 W_lin[C][d][256], V[C][d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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

    W = np.zeros((C, d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    t = 0

    n = X.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(idx)

            grad_W = np.zeros_like(W)
            grad_V = np.zeros_like(V)

            for b in range(bsz):
                x = xb[b]
                y_true = yb[b]

                # Linear scores (C,)
                lin = W[:, np.arange(d), x].sum(axis=1)

                # Selected latent vectors (C, d, K)
                V_sel = V[:, np.arange(d), x, :]
                sum_V = V_sel.sum(axis=1)                 # (C, K)
                sq_sum_V = (sum_V ** 2).sum(axis=1)       # (C,)
                sum_sq_V = (V_sel ** 2).sum(axis=(1, 2))  # (C,)
                inter = 0.5 * (sq_sum_V - sum_sq_V)

                scores = lin + inter
                max_s = scores.max()
                exp_s = np.exp(scores - max_s)
                probs = exp_s / exp_s.sum()

                dscores = probs
                dscores[y_true] -= 1.0  # (C,)

                # Gradient for linear weights
                for j in range(d):
                    grad_W[:, j, x[j]] += dscores

                # Gradient for latent vectors
                # dL/dV_cjk = dscores[c] * (sum_V[c,k] - V_sel[c,j,k])
                for j in range(d):
                    factor = dscores[:, np.newaxis] * (sum_V - V_sel[:, j, :])  # (C, K)
                    grad_V[:, j, x[j], :] += factor

            # Adam update with L2
            t += 1
            # W
            gW = grad_W + L2 * W
            mW = BETA1 * mW + (1 - BETA1) * gW
            vW = BETA2 * vW + (1 - BETA2) * gW * gW
            mW_hat = mW / (1 - BETA1 ** t)
            vW_hat = vW / (1 - BETA2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + EPS)

            # V
            gV = grad_V + L2 * V
            mV = BETA1 * mV + (1 - BETA1) * gV
            vV = BETA2 * vV + (1 - BETA2) * gV * gV
            mV_hat = mV / (1 - BETA1 ** t)
            vV_hat = vV / (1 - BETA2 ** t)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + EPS)

    # Write weights
    magic = 0x464D4C31  # FML1
    header = struct.pack("<4i", magic, d, C, K)
    out = header + W.astype("<f4").tobytes() + V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 16 || C != 65 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t W_size = (size_t)C * d * 256;
    size_t V_size = (size_t)C * d * 256 * K;
    long expected = 16 + 4L * (W_size + V_size);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(wb + 16);
    const float *V = (const float *)(wb + 16 + 4L * W_size);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(K * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            // Linear part
            float lin = 0.0f;
            for (int j = 0; j < d; j++) {
                lin += W[((size_t)c * d + j) * 256 + xi[j]];
            }
            // Interaction part via FM trick
            memset(sum_V, 0, K * sizeof(float));
            float sum_sq_V = 0.0f;
            for (int j = 0; j < d; j++) {
                const float *v_ptr = V + (((size_t)c * d + j) * 256 + xi[j]) * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = v_ptr[kk];
                    sum_V[kk] += val;
                    sum_sq_V += val * val;
                }
            }
            float sq_sum_V = 0.0f;
            for (int kk = 0; kk < K; kk++) sq_sum_V += sum_V[kk] * sum_V[kk];
            float inter = 0.5f * (sq_sum_V - sum_sq_V);
            scores[c] = lin + inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```