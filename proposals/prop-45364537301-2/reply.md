```hypothesis
Factorization Machine (FM) captures pairwise byte-position interactions crucial for character-level language modeling. The linear model (0.3375) treats each position independently, but n-grams like "th", "he", "in" are predictive. FM adds a low-rank interaction term (embedding dim k=4) sharing statistical strength across positions, adding only ~4MB weights. Trained with Adam on cross-entropy, it should exceed 0.36 accuracy at <15k Ir/item. Failure: if dev accuracy ≤0.34 or Ir/item >25k, interactions don't help or implementation is too slow.
```
```python
"""Factorization Machine for next-char prediction: pairwise interactions via low-rank embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy. Exports: magic 0x464D31, d, k, C; float32 linear[d*256][C], V[d*256][C][k]."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    nfeat = d * 256
    # linear weights [nfeat, C]
    W = np.zeros((nfeat, C), np.float32)
    # embeddings [nfeat, C, K]
    V = (rng.normal(0, 0.01, (nfeat, C, K))).astype(np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # active feature indices: (batch, d)
            feat_idx = np.arange(d) * 256 + xb  # (bsz, d)

            # Linear scores: (bsz, C)
            lin = np.zeros((bsz, C), np.float32)
            for j in range(d):
                lin += W[feat_idx[:, j]]

            # Interaction scores
            # s_c = sum_j V[f_j, c]  -> (bsz, C, K)
            s = np.zeros((bsz, C, K), np.float32)
            sum_sq = np.zeros((bsz, C), np.float32)
            for j in range(d):
                v = V[feat_idx[:, j]]  # (bsz, C, K)
                s += v
                sum_sq += np.sum(v * v, axis=2)
            inter = 0.5 * (np.sum(s * s, axis=2) - sum_sq)  # (bsz, C)

            scores = lin + inter  # (bsz, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad = probs / bsz  # (bsz, C)

            # Gradient wrt linear: sum over batch
            gW = np.zeros_like(W)
            for j in range(d):
                np.add.at(gW, feat_idx[:, j], grad)

            # Gradient wrt V
            # dL/dV[f, c, k] = sum_i grad[i,c] * (s[i,c,k] - V[f,c,k]) for active f
            gV = np.zeros_like(V)
            for j in range(d):
                f = feat_idx[:, j]  # (bsz,)
                v = V[f]  # (bsz, C, K)
                # s - v  -> (bsz, C, K)
                diff = s - v
                # grad[:, :, None] * diff  -> (bsz, C, K)
                g = grad[:, :, None] * diff
                np.add.at(gV, f, g)

            # L2 regularization
            gW += L2 * W
            gV += L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * (gV * gV)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes (categorical). Output: class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    int nfeat = d * 256;
    long expected = 16L + 4L * nfeat * C + 4L * nfeat * C * K;
    if (C < 1 || C > 256 || K < 1 || K > 64 || nw != expected) {
        fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3;
    }
    const float *W = (const float *)(hd + 4);
    const float *V = W + (long)nfeat * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *lin = calloc(C, sizeof(float));
    float *s = calloc((long)C * K, sizeof(float));
    float *sum_sq = calloc(C, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(lin, 0, C * sizeof(float));
        memset(s, 0, (long)C * K * sizeof(float));
        memset(sum_sq, 0, C * sizeof(float));

        for (int j = 0; j < d; j++) {
            int f = j * 256 + xi[j];
            const float *w_f = W + (long)f * C;
            const float *v_f = V + (long)f * C * K;
            for (int c = 0; c < C; c++) {
                lin[c] += w_f[c];
                float *s_c = s + (long)c * K;
                for (int kk = 0; kk < K; kk++) {
                    float v = v_f[(long)c * K + kk];
                    s_c[kk] += v;
                    sum_sq[c] += v * v;
                }
            }
        }

        float best_score = -1e30f;
        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            float *s_c = s + (long)c * K;
            for (int kk = 0; kk < K; kk++) inter += s_c[kk] * s_c[kk];
            inter = 0.5f * (inter - sum_sq[c]);
            float score = lin[c] + inter;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```