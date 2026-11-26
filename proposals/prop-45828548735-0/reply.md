```hypothesis
Factorization Machine (FM) with tiny embeddings (k=4) adds pairwise feature interactions at near-linear cost. Adult is known to have strong interactions (education×occupation, marital×occupation) that linear models miss; FM captures them with O(d·k) extra parameters and ~d·k FLOPs per item. If dev accuracy does not exceed label-linear (0.8524) by ≥0.5 pp (i.e. <0.857), the interaction signal is too weak for this representation.
```
```python
"""Factorization Machine family for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: magic 0x464D3132, int32 d,k,C; float32 linear[d*256*C], emb[d*256*k], bias[C], v_weight[k*C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 4  # embedding size


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters
    linear = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    v_weight = (rng.normal(0, 0.01, (K, C))).astype(np.float32)
    params = [linear, emb, bias, v_weight]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            lin_part = np.zeros((bsz, C), np.float32)
            sum_emb = np.zeros((bsz, K), np.float32)
            sum_sq = np.zeros((bsz, K), np.float32)
            for j in range(d):
                v = xb[:, j]
                lin_part += linear[j, v]
                e = emb[j, v]            # (bsz, K)
                sum_emb += e
                sum_sq += e * e
            inter = 0.5 * (sum_emb * sum_emb - sum_sq)          # (bsz, K)
            scores = bias + lin_part + inter @ v_weight         # (bsz, C)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_scores = (probs / bsz).astype(np.float32)      # (bsz, C)

            # gradients
            g_bias = grad_scores.sum(axis=0)
            g_v_weight = inter.T @ grad_scores                  # (K, C)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            # backprop through interaction
            g_inter = grad_scores @ v_weight.T                  # (bsz, K)
            # d(inter)/d(sum_emb) = sum_emb, d(inter)/d(sum_sq) = -0.5
            g_sum_emb = g_inter * sum_emb
            g_sum_sq = g_inter * (-0.5)

            for j in range(d):
                v = xb[:, j]
                # linear gradient
                np.add.at(g_lin[j], v, grad_scores)
                # embedding gradient
                e = emb[j, v]
                # from sum_emb
                np.add.at(g_emb[j], v, g_sum_emb)
                # from sum_sq: d(sum_sq)/de = 2*e
                np.add.at(g_emb[j], v, g_sum_sq * (2 * e))

            # L2 regularization on all but bias
            g_lin += L2 * linear
            g_emb += L2 * emb
            g_v_weight += L2 * v_weight

            grads = [g_lin, g_emb, g_bias, g_v_weight]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x464D3132
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += v_weight.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }

    size_t lin_sz = (size_t)d * 256 * C;
    size_t emb_sz = (size_t)d * 256 * K;
    size_t bias_sz = C;
    size_t vw_sz = (size_t)K * C;
    size_t total_floats = lin_sz + emb_sz + bias_sz + vw_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *emb = lin + lin_sz;
    const float *bias = emb + emb_sz;
    const float *v_weight = bias + bias_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int f = 0; f < K; f++) { sum_emb[f] = 0.0f; sum_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_jv = lin + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_jv[c];
            const float *emb_jv = emb + ((size_t)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float e = emb_jv[f];
                sum_emb[f] += e;
                sum_sq[f] += e * e;
            }
        }
        for (int f = 0; f < K; f++) {
            float inter = 0.5f * (sum_emb[f] * sum_emb[f] - sum_sq[f]);
            for (int c = 0; c < C; c++) scores[c] += inter * v_weight[(size_t)f * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```