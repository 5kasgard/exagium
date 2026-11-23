```hypothesis
Family: label-fm (Factorization Machine with 8-dimensional embeddings).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital×sex) that linear models miss. A rank-8 FM captures these with O(d·k) parameters (d=14, k=8) and adds only ~14×256×8 ≈ 28k floats. Training with Adam on binary cross-entropy is fast and deterministic. Inference is two vector sums and a dot product per record, keeping CPU cost near label-linear.
Failure sign: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, k, C; float32 linear[d*256], emb[d*256*k], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-5, 256, 10


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
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            lin_part = np.zeros(bsz, np.float32)
            emb_sum = np.zeros((bsz, K), np.float32)
            emb_sq_sum = np.zeros((bsz, K), np.float32)
            for j in range(d):
                vals = xb[:, j]
                lin_part += linear[j, vals]
                e = emb[j, vals]                 # (bsz, K)
                emb_sum += e
                emb_sq_sum += e * e
            inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
            logit = lin_part + inter + bias[1]   # class 1 logit (class 0 logit = bias[0])
            # binary cross-entropy gradient wrt logit
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32) / bsz

            # backward
            # bias
            g_bias = np.zeros(C, np.float32)
            g_bias[1] = grad.sum()
            # linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad)
            # emb
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                # d inter / d emb = (emb_sum - e) * grad[:, None]
                diff = (emb_sum - emb[j, vals]) * grad[:, None]
                np.add.at(g_emb[j], vals, diff)

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for param, grad, m, v in [
                (linear, g_lin + L2 * linear, m_lin, v_lin),
                (emb, g_emb + L2 * emb, m_emb, v_emb),
                (bias, g_bias, m_bias, v_bias),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weights layout: magic(0x464D3130), d, k, C; float32 linear[d*256], emb[d*256*k], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t lin_sz = (size_t)d * 256;
    size_t emb_sz = (size_t)d * 256 * K;
    size_t bias_sz = C;
    size_t expected = 16 + 4 * (lin_sz + emb_sz + bias_sz);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + lin_sz;
    const float *bias = emb + emb_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_part = 0.0f;
        for (int f = 0; f < K; f++) { emb_sum[f] = 0.0f; emb_sq_sum[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            lin_part += linear[(size_t)j * 256 + v];
            const float *e = emb + ((size_t)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                emb_sum[f] += val;
                emb_sq_sum[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float s = emb_sum[f];
            inter += s * s - emb_sq_sum[f];
        }
        inter *= 0.5f;
        float logit = lin_part + inter + bias[1];  // class 1 logit; class 0 logit = bias[0]
        unsigned char pred = (logit > bias[0]) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```