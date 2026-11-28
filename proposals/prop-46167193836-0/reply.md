```hypothesis
Factorization Machine (FM) with shared latent vectors (rank 8) adds pairwise feature interactions
to a linear model. On tabular data with mixed ordinal/categorical features, FMs often outperform
pure linear models at modest extra cost (O(d·k) per item). Here d=14, k=8 gives ~150 FLOPs/item,
far cheaper than an MLP. If dev accuracy does not exceed label-linear (0.8524), the interactions
captured are either irrelevant or the rank is too low.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Model: logit_c = bias_c + sum_j W_j[x_j]_c + w_int_c * (0.5*( (sum_j V_j[x_j])^2 - sum_j V_j[x_j]^2 ))
Exported as int32 magic 0x464D4C31, d, C, k; float32 W[d][256][C], bias[C], V[d][256][k], w_int[C]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # linear embeddings per feature, per byte, per class
    W = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    # latent vectors per feature, per byte, per latent dim
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    w_int = np.zeros(C, np.float32)

    params = [W, bias, V, w_int]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (bs, d)
            yb = y[idx]          # (bs,)
            bs = xb.shape[0]

            # linear term: (bs, C)
            lin = bias + np.sum(W[np.arange(d), xb.T, :], axis=0).T   # W[d, xb[:,j], :] summed over j

            # latent sum and sum of squares: (bs, K)
            V_sel = V[np.arange(d), xb.T, :]          # (d, bs, K)
            sum_v = np.sum(V_sel, axis=0)             # (bs, K)
            sum_sq = np.sum(V_sel * V_sel, axis=0)    # (bs, K)

            # interaction scalar per sample
            inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(sum_sq, axis=1))  # (bs,)

            # logits: (bs, C)
            logits = lin + w_int * inter[:, None]

            # softmax + cross-entropy gradient
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)   # (bs, C)

            # gradients
            # bias
            g_bias = grad_logits.sum(axis=0)
            # w_int
            g_w_int = (grad_logits * inter[:, None]).sum(axis=0)
            # W: (d, 256, C)
            g_W = np.zeros_like(W)
            for j in range(d):
                # xb[:, j] is (bs,)
                np.add.at(g_W[j], xb[:, j], grad_logits)
            # V: (d, 256, K)
            g_V = np.zeros_like(V)
            # d/dV of inter = sum_v - V_sel  (per feature, per sample)
            # grad_logits[:, c] * w_int[c] gives weight per sample for interaction
            inter_weight = (grad_logits @ w_int).astype(np.float32)  # (bs,)
            for j in range(d):
                # for each sample, the gradient for V[j, xb[:,j], :] is inter_weight * (sum_v - V_sel[j])
                delta = (inter_weight[:, None] * (sum_v - V_sel[j])).astype(np.float32)  # (bs, K)
                np.add.at(g_V[j], xb[:, j], delta)

            grads = [g_W, g_bias, g_V, g_w_int]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # L2 regularization on W and V only
                if i == 0 or i == 2:
                    p -= LR * L2 * p

    # pack weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, C, K)
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += w_int.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad header values\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)
        + (size_t)C * sizeof(float)
        + (size_t)d * 256 * K * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W = p; p += (size_t)d * 256 * C;
    const float *bias = p; p += C;
    const float *V = p; p += (size_t)d * 256 * K;
    const float *w_int = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    float *sum_v = malloc(K * sizeof(float));
    float *sum_sq = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_sq[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Wj = W + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) logits[c] += Wj[c];

            const float *Vj = V + ((size_t)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                sum_v[kk] += val;
                sum_sq[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_sq[kk];
        inter *= 0.5f;

        for (int c = 0; c < C; c++) logits[c] += w_int[c] * inter;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```