```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via low-rank factorization (K=8), adding only O(d*K) params and O(d*K) inference cost. If dev accuracy ≤ 0.8524 (label-linear), interactions don't help or K is too small.
```
```python
"""Factorization Machine for tabular label tasks: FM with K=8 factors.
python3 -s -P train.py card.json train.in train.key weights.bin (deterministic: seed 1, single thread)
Encoding per card types: ordinal z-scored, categorical one-hot over training values.
FM: logit = w0 + sum w_i x_i + 0.5 sum_f [(sum_i v_{if} x_i)^2 - sum_i (v_{if} x_i)^2].
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: magic 0x464D31, d, K, C; float32 T[d][256][1+K], bias."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    assert C == 2
    rng = np.random.default_rng(SEED)

    # Encode features same as label-linear
    enc = []
    feature_col = []
    col_feature_start = []
    col_feature_end = []
    feature_is_ordinal = []
    ordinal_params = []
    categorical_vals = []

    feat_idx = 0
    for j, t in enumerate(card["input"]["types"]):
        col_feature_start.append(feat_idx)
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_params.append((mu, sd))
            categorical_vals.append(None)
            feature_is_ordinal.append(True)
            feat_idx += 1
        else:
            vals = np.unique(X[:, j])
            categorical_vals.append(vals)
            ordinal_params.append(None)
            feature_is_ordinal.append(False)
            feat_idx += len(vals)
        col_feature_end.append(feat_idx)
        for f in range(col_feature_start[j], col_feature_end[j]):
            feature_col.append(j)

    n_features = feat_idx
    Z = np.zeros((X.shape[0], n_features), dtype=np.float32)
    for j in range(d):
        start, end = col_feature_start[j], col_feature_end[j]
        if feature_is_ordinal[j]:
            mu, sd = ordinal_params[j]
            Z[:, start] = (X[:, j].astype(np.float32) - mu) / sd
        else:
            vals = categorical_vals[j]
            for idx, val in enumerate(vals):
                Z[X[:, j] == val, start + idx] = 1.0

    n, k = Z.shape
    assert k == n_features

    # FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(k, dtype=np.float32)
    V = (rng.uniform(-0.01, 0.01, (k, K)) * math.sqrt(6.0 / k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]

            # Forward
            linear = w0 + z @ w
            sum_vx = z @ V
            sum_v2x2 = (z ** 2) @ (V ** 2)
            fm = linear + 0.5 * np.sum(sum_vx ** 2 - sum_v2x2, axis=1)

            # Sigmoid and gradient
            p = 1.0 / (1.0 + np.exp(-fm))
            grad = (p - yy).astype(np.float32)

            # Gradients
            gw0 = grad.mean()
            gw = (grad[:, None] * z).mean(axis=0) + L2 * w
            z_exp = z[:, :, None]
            sum_vx_exp = sum_vx[:, None, :]
            term1 = z_exp * sum_vx_exp
            term2 = (z_exp ** 2) * V[None, :, :]
            gV = (grad[:, None, None] * (term1 - term2)).mean(axis=0) + L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Export weights as lookup tables T[d][256][1+K]
    T = np.zeros((d, 256, 1 + K), dtype=np.float32)
    for j in range(d):
        start, end = col_feature_start[j], col_feature_end[j]
        if feature_is_ordinal[j]:
            mu, sd = ordinal_params[j]
            for v in range(256):
                x = (v - mu) / sd
                T[j, v, 0] = w[start] * x
                T[j, v, 1:] = V[start] * x
        else:
            vals = categorical_vals[j]
            for idx, val in enumerate(vals):
                feat = start + idx
                T[j, val, 0] = w[feat]
                T[j, val, 1:] = V[feat]

    bias = np.array([w0], dtype=np.float32)
    magic = 0x464D31  # 'FM1'
    out = struct.pack("<4i", magic, d, K, C)
    out += T.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (0 or 1).
 * Weights format: magic 0x464D31, d, K, C; float32 T[d][256][1+K]; float32 bias. */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * d * 256 * (1 + K) + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * (1 + K);
    float b = bias[0];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vx = calloc(K, sizeof(float));
    float *sum_v2x2 = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = b;
        for (int f = 0; f < K; f++) { sum_vx[f] = 0.0f; sum_v2x2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * (1 + K);
            linear += t[0];
            for (int f = 0; f < K; f++) {
                float v = t[1 + f];
                sum_vx[f] += v;
                sum_v2x2[f] += v * v;
            }
        }

        float fm = linear;
        for (int f = 0; f < K; f++) {
            fm += 0.5f * (sum_vx[f] * sum_vx[f] - sum_v2x2[f]);
        }

        unsigned char pred = (fm > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_vx);
    free(sum_v2x2);
    return 0;
}
```