```hypothesis
Propose label-fm: a factorization machine with 4-dimensional embeddings per feature value.
Adult income has strong feature interactions (e.g., education×occupation); linear models miss these.
FM captures all pairwise interactions efficiently via the (sum^2 - sum of squares) trick.
If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, k, C; float32 linear[d][256], embed[d][256][k], bias."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 15


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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: linear[d][256], embed[d][256][K], bias
    linear = [rng.normal(0, 0.01, 256).astype(np.float32) for _ in range(d)]
    embed  = [rng.normal(0, 0.01, (256, K)).astype(np.float32) for _ in range(d)]
    bias = np.float32(0.0)

    # Adam state
    m_lin = [np.zeros_like(w) for w in linear]
    v_lin = [np.zeros_like(w) for w in linear]
    m_emb = [np.zeros_like(e) for e in embed]
    v_emb = [np.zeros_like(e) for e in embed]
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            # Forward
            lin_sum = np.zeros(B, np.float32)
            sum_vec = np.zeros((B, K), np.float32)
            sum_sq  = np.zeros((B, K), np.float32)
            for j in range(d):
                vals = xb[:, j]
                lin_sum += linear[j][vals]
                ev = embed[j][vals]
                sum_vec += ev
                sum_sq  += ev * ev

            inter = 0.5 * np.sum(sum_vec * sum_vec - sum_sq, axis=1)
            logit = bias + lin_sum + inter
            # Logistic loss gradient (y in {0,1})
            p = 1.0 / (1.0 + np.exp(-logit))
            dlogit = p - yb  # (B,)

            # Backward
            db = np.mean(dlogit)
            # Gradients for linear and embed
            g_lin = [np.zeros(256, np.float32) for _ in range(d)]
            g_emb = [np.zeros((256, K), np.float32) for _ in range(d)]
            for j in range(d):
                vals = xb[:, j]
                # linear gradient
                np.add.at(g_lin[j], vals, dlogit)
                # embedding gradient: dlogit * (sum_vec - ev)
                ev = embed[j][vals]
                for f in range(K):
                    grad_f = dlogit * (sum_vec[:, f] - ev[:, f])
                    np.add.at(g_emb[j][:, f], vals, grad_f)

            # Adam update
            t += 1
            # bias
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db * db
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

            for j in range(d):
                # linear
                gl = g_lin[j] + 2 * L2 * linear[j]
                m_lin[j] = beta1 * m_lin[j] + (1 - beta1) * gl
                v_lin[j] = beta2 * v_lin[j] + (1 - beta2) * gl * gl
                m_hat = m_lin[j] / (1 - beta1 ** t)
                v_hat = v_lin[j] / (1 - beta2 ** t)
                linear[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)

                # embed
                ge = g_emb[j] + 2 * L2 * embed[j]
                m_emb[j] = beta1 * m_emb[j] + (1 - beta1) * ge
                v_emb[j] = beta2 * v_emb[j] + (1 - beta2) * ge * ge
                m_hat = m_emb[j] / (1 - beta1 ** t)
                v_hat = v_emb[j] / (1 - beta2 ** t)
                embed[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Header: magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    for j in range(d):
        out += linear[j].astype("<f4").tobytes()
        out += embed[j].astype("<f4").tobytes()
    out += np.array([bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N */
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
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4L * (d * 256L * (1L + K) + 1);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *linear[d];
    const float *embed[d];
    for (int j = 0; j < d; j++) {
        linear[j] = p; p += 256;
        embed[j] = p; p += 256 * K;
    }
    float bias = *p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_vec[K];
    float sum_sq[K];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = 0.0f;
        for (int f = 0; f < K; f++) { sum_vec[f] = 0.0f; sum_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            lin_sum += linear[j][v];
            const float *ev = embed[j] + (size_t)v * K;
            for (int f = 0; f < K; f++) {
                float val = ev[f];
                sum_vec[f] += val;
                sum_sq[f]  += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sum_vec[f] * sum_vec[f] - sum_sq[f];
        inter *= 0.5f;

        float logit = bias + lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```