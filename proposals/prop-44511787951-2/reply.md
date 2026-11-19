```hypothesis
Family: label-fm (factorization machine with latent dimension 8).
Why: Adult income classification benefits from pairwise feature interactions (e.g., education × occupation). A factorization machine adds these interactions with only O(d·k) extra parameters (d=14, k=8) and O(d·k) inference cost, keeping it fast. It should surpass label-linear (0.8524) and approach label-mlp16 (0.8591) while being cheaper than label-gbdt.
Failure: If dev accuracy ≤ 0.8524 or inference instructions/item exceed 2× label-linear.
```

```python
"""Factorization machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 50


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

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            linear = w0 + w[np.arange(d)[:, None], xb.T].sum(axis=0)  # (bsz,)
            # interaction
            V_sel = V[np.arange(d)[:, None], xb.T]  # (d, bsz, K)
            sum_V = V_sel.sum(axis=0)                # (bsz, K)
            sum_V2 = (V_sel * V_sel).sum(axis=0)     # (bsz, K)
            inter = 0.5 * ((sum_V * sum_V - sum_V2).sum(axis=1))  # (bsz,)
            score = linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - yb).astype(np.float32)  # (bsz,)

            # gradients
            gw0 = grad_score.mean()
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)

            # linear gradients
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gw[j], vals, grad_score)
            gw /= bsz

            # interaction gradients
            # d inter / d V[j, v, f] = (p-y) * (sum_V[f] - V[j, v, f])
            for j in range(d):
                vals = xb[:, j]
                for f in range(K):
                    sVf = sum_V[:, f]
                    Vf = V_sel[j, :, f]
                    g = grad_score * (sVf - Vf)
                    np.add.at(gV[j, :, f], vals, g)
            gV /= bsz

            # L2 regularization
            gw += L2 * w
            gV += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # write weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization machine inference for label tasks: ./infer weights.bin in.rec N
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
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[8], sum_V2[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) score += w[(long)j * 256 + xi[j]];

        for (int f = 0; f < K; f++) { sum_V[f] = 0.0f; sum_V2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = vj[f];
                sum_V[f] += v;
                sum_V2[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) score += 0.5f * (sum_V[f] * sum_V[f] - sum_V2[f]);

        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```