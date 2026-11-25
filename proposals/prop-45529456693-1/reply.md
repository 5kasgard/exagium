```hypothesis
Family: label-fm (factorization machine with rank 8). The 64 ordinal pixel counts are z-scored per feature, then fed into a factorization machine that captures pairwise feature interactions via a low-rank decomposition. This should outperform linear models on optdigits because digit shapes involve correlated pixel patterns (strokes, loops) that are multiplicative in nature. If dev accuracy does not exceed 0.9613 (label-linear), the interactions are not beneficial or the rank is insufficient.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each byte as z-scored ordinal. Trains a factorization machine of rank K=8
with softmax cross-entropy, Adam (lr=3e-3, L2=1e-4), minibatch 64, epochs min(100, ceil(600000/n)).
Exports: magic 0x464D4C31, d, K, C; float32 mean[d], std[d], w0[C], w[C*d], V[C*d*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8  # FM rank


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

    # z-score normalization per feature
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).astype(np.float32)

    # parameters
    w0 = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (C, d))).astype(np.float32)
    V = (rng.normal(0, 0.01, (C, d, K))).astype(np.float32)
    params = [w0, w, V]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X_norm.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_norm[idx]
            yb = y[idx]
            B = len(yb)

            # forward
            linear = Xb @ w.T + w0  # (B, C)
            S_vx = np.einsum('bd,cdk->bck', Xb, V)          # (B, C, K)
            S_vx2 = np.einsum('bd,cdk->bck', Xb**2, V**2)    # (B, C, K)
            interaction = 0.5 * np.sum(S_vx**2 - S_vx2, axis=2)  # (B, C)
            scores = linear + interaction

            # softmax
            max_score = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # gradient w.r.t scores
            dscores = probs
            dscores[np.arange(B), yb] -= 1.0
            dscores /= B

            # gradients
            dw0 = dscores.sum(axis=0)
            dw = dscores.T @ Xb + L2 * w

            A = np.einsum('bc,bi,bj->cij', dscores, Xb, Xb)   # (C, d, d)
            B_mat = np.einsum('bc,bj->cj', dscores, Xb**2)     # (C, d)
            dV = np.zeros_like(V)
            for f in range(K):
                V_f = V[:, :, f]
                dV_f = np.einsum('cij,ci->cj', A, V_f) - V_f * B_mat
                dV[:, :, f] = dV_f + L2 * V_f

            grads = [dw0, dw, dV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=64). Output: one byte class index (0..9). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64 || K != 8) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16
        + (long)d * 4 * 2          // mean, std
        + (long)C * 4              // w0
        + (long)C * d * 4          // w
        + (long)C * d * K * 4;     // V
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *w0   = std + d;
    const float *w    = w0 + C;
    const float *V    = w + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float x_norm[64];
    float scores[10];
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            x_norm[j] = (xi[j] - mean[j]) / std[j];
        }

        for (int c = 0; c < C; c++) {
            float sc = w0[c];
            for (int j = 0; j < d; j++) sc += w[c * d + j] * x_norm[j];
            for (int f = 0; f < K; f++) {
                float sum_vx = 0.0f, sum_v2x2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = V[((long)c * d + j) * K + f];
                    float xv = v * x_norm[j];
                    sum_vx += xv;
                    sum_v2x2 += v * xv;  // v * (v * x_norm[j]) = v^2 * x_norm[j]
                }
                sc += 0.5f * (sum_vx * sum_vx - sum_v2x2);
            }
            scores[c] = sc;
        }

        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```