```hypothesis
Family: label-fm (Factorization Machine). The Adult task has 14 mixed ordinal/categorical features where pairwise interactions (e.g., education×occupation, marital×relationship) are known to be predictive. A degree-2 FM captures all pairwise interactions via k-dimensional embeddings per (feature,value) with O(d·k) inference cost, far cheaper than an MLP. With k=8 it adds ~36K params vs ~7K for linear, should beat 0.8524 accuracy while staying near 719 Ir/item. Failure: dev accuracy ≤ 0.8524 or Ir/item > 1000.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-byte lookup tables for linear + FM embeddings. FM rank k=8.
Adam lr 3e-3, L2 1e-4 (linear), L2 1e-3 (FM), minibatch 64, epochs min(100, ceil(600000/n)).
Weights: magic 0x464D3132, d, k, C; float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][k], V[k][C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_LIN, L2_FM, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-3, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Build encoders (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Linear weights: d x 256 x C (via encoders)
    lin_shapes = [E.shape[1] for E in enc]
    lin_total = sum(lin_shapes)
    W_lin = (rng.uniform(-1, 1, (lin_total, C)) * math.sqrt(6.0 / lin_total)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)

    # FM embeddings: d x 256 x K (via encoders)
    fm_total = lin_total  # same encoder dimensions
    W_fm = (rng.normal(0, 0.01, (fm_total, K))).astype(np.float32)

    # FM projection: K x C
    V = (rng.normal(0, 0.01, (K, C))).astype(np.float32)

    params = [W_lin, b_lin, W_fm, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute encoded features
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z_lin = np.concatenate(Z_list, axis=1).astype(np.float32)  # n x lin_total
    Z_fm = Z_lin  # same for FM

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z_lin = Z_lin[idx]
            z_fm = Z_fm[idx]
            yy = y[idx]
            m = len(yy)

            # Forward
            logits = z_lin @ W_lin + b_lin  # m x C

            # FM part: 0.5 * ((sum v_i)^2 - sum v_i^2) @ V
            sum_v = z_fm @ W_fm  # m x K
            sum_v2 = (z_fm ** 2) @ (W_fm ** 2)  # m x K
            fm_interaction = 0.5 * (sum_v ** 2 - sum_v2)  # m x K
            logits += fm_interaction @ V  # m x C

            # Softmax + cross-entropy gradient
            max_logit = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_logits = (probs / m).astype(np.float32)  # m x C

            # Gradients
            grad_W_lin = z_lin.T @ grad_logits + L2_LIN * W_lin
            grad_b_lin = grad_logits.sum(axis=0)

            # FM gradients
            # dL/d(fm_interaction) = grad_logits @ V.T  (m x K)
            grad_fm_inter = grad_logits @ V.T  # m x K
            # d(fm_inter)/d(sum_v) = sum_v
            grad_sum_v = grad_fm_inter * sum_v  # m x K
            # d(fm_inter)/d(sum_v2) = -0.5
            grad_sum_v2 = -0.5 * grad_fm_inter  # m x K

            grad_W_fm = z_fm.T @ grad_sum_v + (z_fm ** 2).T @ grad_sum_v2 + L2_FM * W_fm
            grad_V = fm_interaction.T @ grad_logits

            grads = [grad_W_lin, grad_b_lin, grad_W_fm, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                p -= LR * (m1 / (1 - 0.9 ** t)) / (np.sqrt(v1 / (1 - 0.999 ** t)) + 1e-8)

    # Build lookup tables T_lin[d][256][C], T_fm[d][256][K]
    T_lin = np.zeros((d, 256, C), np.float32)
    T_fm = np.zeros((d, 256, K), np.float32)
    off = 0
    for j, E in enumerate(enc):
        nvals = E.shape[1]
        T_lin[j] = (E.astype(np.float32) @ W_lin[off:off + nvals]).astype(np.float32)
        T_fm[j] = (E.astype(np.float32) @ W_fm[off:off + nvals]).astype(np.float32)
        off += nvals

    # Pack weights
    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += T_fm.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class.
 * Weights format: magic 0x464D3132, d, k, C; float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][k], V[k][C]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16L +
        4L * d * 256 * C + 4L * C +
        4L * d * 256 * K + 4L * K * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *b_lin = T_lin + (long)d * 256 * C;
    const float *T_fm = b_lin + C;
    const float *V = T_fm + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b_lin[c];
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v2[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            const float *tlin = T_lin + ((long)j * 256 + xi[j]) * C;
            const float *tfm = T_fm + ((long)j * 256 + xi[j]) * K;
            for (int c = 0; c < C; c++) logits[c] += tlin[c];
            for (int u = 0; u < K; u++) {
                float v = tfm[u];
                sum_v[u] += v;
                sum_v2[u] += v * v;
            }
        }

        for (int u = 0; u < K; u++) {
            float inter = 0.5f * (sum_v[u] * sum_v[u] - sum_v2[u]);
            for (int c = 0; c < C; c++) logits[c] += inter * V[(long)u * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```