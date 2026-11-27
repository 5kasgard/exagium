```hypothesis
Factorization Machine (FM) with latent dimension k=4, capturing pairwise feature interactions for the 14 byte-valued features. FMs are strong on tabular data with categorical features. Model size ~36K params (two-class). Inference cost O(d*k) per item, similar to linear. Expect dev accuracy >0.8524 (label-linear) with comparable ns/item. Failure: accuracy ≤ label-linear or cost >2× label-linear.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread, numpy only.
Encoding: each of d bytes is a categorical feature with 256 values.
Model: for each class c, bias b_c, linear weights W_c[d,256], embedding V_c[d,256,k].
Score_c = b_c + sum_j W_c[j, x_j] + 0.5 * sum_f ( (sum_j V_c[j,x_j,f])^2 - sum_j V_c[j,x_j,f]^2 ).
Adam lr=0.01, L2=1e-4, batch=64, epochs min(100, ceil(600000/n)).
Export: magic 0x464D4C31, d, k, C; float32 W[C,d,256], V[C,d,256,k], b[C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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

    # parameters: W[C, d, 256], V[C, d, 256, K], b[C]
    W = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            # linear part: (bs, C)
            lin = b + np.sum(W[:, xb.T, np.arange(d)], axis=1).T  # W[:, xb.T, :] -> (C, d, bs) sum over d -> (C, bs) -> (bs, C)
            # interaction part
            # V_gather: (C, d, bs, K)
            Vg = V[:, xb.T, np.arange(d), :]  # (C, d, bs, K)
            sumV = Vg.sum(axis=1)  # (C, bs, K)
            sqV = (Vg * Vg).sum(axis=1)  # (C, bs, K)
            inter = 0.5 * ((sumV * sumV - sqV).sum(axis=2)).T  # (bs, C)
            scores = lin + inter  # (bs, C)

            # softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_scores = (probs / bs).astype(np.float32)  # (bs, C)

            # gradients
            # linear grads
            gW = np.zeros_like(W)
            gb = grad_scores.sum(axis=0)  # (C,)
            # for each sample and feature, add grad_scores to W[c, j, x_j]
            # vectorized: (bs, d) -> for each class c, we need to accumulate over samples
            # We'll loop over classes (C=2) and features (d=14) for simplicity; bs up to 64.
            for c in range(C):
                for j in range(d):
                    vals = xb[:, j]  # (bs,)
                    np.add.at(gW[c, j], vals, grad_scores[:, c])

            # interaction grads
            gV = np.zeros_like(V)
            # For each class c, feature j, latent f:
            # d inter / d V[c,j,x_j,f] = (sumV[c,:,f] - V[c,j,x_j,f]) * grad_scores[:,c]
            for c in range(C):
                for j in range(d):
                    vals = xb[:, j]  # (bs,)
                    Vgj = Vg[c, j]  # (bs, K)
                    sumV_c = sumV[c]  # (bs, K)
                    for f in range(K):
                        coeff = (sumV_c[:, f] - Vgj[:, f]) * grad_scores[:, c]  # (bs,)
                        np.add.at(gV[c, j, :, f], vals, coeff)

            # L2 regularization
            gW += L2 * W
            gV += L2 * V

            # Adam update
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (V, gV, mV, vV), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }

    // Expected size: 16 + 4*(C*d*256 + C*d*256*K + C)
    long expected = 16 + 4L * ((long)C * d * 256 + (long)C * d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W = p; p += (long)C * d * 256;
    const float *V = p; p += (long)C * d * 256 * K;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize with bias
        scores[0] = b[0]; scores[1] = b[1];

        // Linear part
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += W[((long)c * d + j) * 256 + v];
            }
        }

        // Interaction part
        for (int c = 0; c < C; c++) {
            float sumV[4] = {0}, sqV[4] = {0};
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                const float *Vptr = V + (((long)c * d + j) * 256 + v) * K;
                for (int f = 0; f < K; f++) {
                    float val = Vptr[f];
                    sumV[f] += val;
                    sqV[f] += val * val;
                }
            }
            for (int f = 0; f < K; f++) {
                scores[c] += 0.5f * (sumV[f] * sumV[f] - sqV[f]);
            }
        }

        // Argmax
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```