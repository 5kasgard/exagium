```hypothesis
Family: label-fm (Factorization Machine with 8-dimensional embeddings).
Why: Adult has mixed ordinal/categorical features; FM captures pairwise interactions with O(d·D) inference cost (d=14, D=8), cheaper than MLP/GBDT. Ordinal bins and categorical indices are both treated as sparse one-hot features; the low-rank interaction matrix models cross-feature effects that linear models miss.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 1500 (≈2× linear), the family does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Exports: magic, d, D, C, bias, linear[d*256], emb[d*256*D] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

D = 8                 # embedding dimension
SEED = 1
LR = 0.05
L2 = 1e-5
BATCH = 128
EPOCHS = 20
MAGIC = 0x464D3130    # 'FM10'

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
    assert C == 2, "FM binary only"

    rng = np.random.default_rng(SEED)
    # parameters: bias, linear[d,256], emb[d,256,D]
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)

    n = X.shape[0]
    steps_per_epoch = math.ceil(n / BATCH)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsz = len(yb)

            # ---- forward ----
            # linear part
            lin_part = np.zeros(bsz, np.float32)
            for j in range(d):
                lin_part += linear[j, xb[:, j]]
            # interaction part via 0.5*( (sum v)^2 - sum v^2 )
            sum_emb = np.zeros((bsz, D), np.float32)
            sq_emb  = np.zeros((bsz, D), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]        # (b, D)
                sum_emb += e
                sq_emb  += e * e
            inter = 0.5 * np.sum(sum_emb * sum_emb - sq_emb, axis=1)
            score = bias + lin_part + inter
            # sigmoid + BCE gradient
            p = 1.0 / (1.0 + np.exp(-score))
            grad = (p - yb).astype(np.float32) / bsz   # (b,)

            # ---- backward ----
            # bias
            g_bias = grad.sum()
            # linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            # embeddings
            g_emb = np.zeros_like(emb)
            for j in range(d):
                e_j = emb[j, xb[:, j]]                 # (b, D)
                # derivative of inter wrt e_j = (sum_emb - e_j)
                grad_e = grad[:, None] * (sum_emb - e_j)   # (b, D)
                np.add.at(g_emb[j], xb[:, j], grad_e)

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # ---- write weights ----
    out = struct.pack("<4i", MAGIC, d, D, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || D <= 0 || d > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256 + 4L * d * 256 * D;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * D);
    float *sq_emb  = malloc(sizeof(float) * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        memset(sum_emb, 0, D * sizeof(float));
        memset(sq_emb, 0, D * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * D;
            for (int u = 0; u < D; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sq_emb[u]  += val * val;
            }
        }
        for (int u = 0; u < D; u++) {
            score += 0.5f * (sum_emb[u] * sum_emb[u] - sq_emb[u]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```