```hypothesis
Propose label-fm: a factorization machine with latent dimension k=4 on the 14 raw byte fields.
Adult has strong pairwise interactions (e.g., education × occupation) that linear models miss.
A tiny FM adds only ~18k parameters, trains fast with Adam, and infers in ~91·k multiply-adds.
If dev accuracy ≤ label-linear (0.8524) or latency > 2× label-linear, the family fails.
```

```python
"""label-fm: Factorization Machine for tabular byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format: magic 0x464D4C31, int32 d,k,C; float32 bias, W[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4  # latent dimension


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    n, d = X.shape
    assert d == 14
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    W = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = X[batch_idx]          # (B, d)
            yb = y[batch_idx]          # (B,)
            B = len(batch_idx)

            # Forward
            lin = bias + np.sum(W[np.arange(d), xb.T], axis=0)  # (B,)
            inter = np.zeros(B, dtype=np.float32)
            for i in range(d):
                vi = V[i, xb[:, i]]           # (B, K)
                for j in range(i + 1, d):
                    vj = V[j, xb[:, j]]
                    inter += np.sum(vi * vj, axis=1)
            score = lin + inter
            p = sigmoid(score)
            grad = (p - yb) / B               # (B,)

            # Gradients
            g_bias = np.sum(grad)

            gW = np.zeros_like(W)
            gV = np.zeros_like(V)

            for i in range(d):
                # linear weight gradient
                vals = xb[:, i]
                np.add.at(gW[i], vals, grad)

                # interaction gradients
                vi = V[i, vals]               # (B, K)
                for j in range(i + 1, d):
                    vj = V[j, xb[:, j]]
                    gij = grad[:, None] * vj  # (B, K)
                    np.add.at(gV[i], vals, gij)
                    gji = grad[:, None] * vi
                    np.add.at(gV[j], xb[:, j], gji)

            # L2 regularization
            gW += L2 * W
            gV += L2 * V

            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # W
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            W -= LR * (mW / (1 - beta1 ** t)) / (np.sqrt(vW / (1 - beta2 ** t)) + eps)

            # V
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV * gV
            V -= LR * (mV / (1 - beta1 ** t)) / (np.sqrt(vV / (1 - beta2 ** t)) + eps)

    # Write weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    blob = header
    blob += np.array([bias], dtype="<f4").tobytes()
    blob += W.astype("<f4").tobytes()
    blob += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d=14 byte records. Output: one byte per record (0 or 1). */
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
    if (d != 14 || C != 2 || K < 1 || K > 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        for (int j = 0; j < d; j++) score += W[(long)j * 256 + xi[j]];

        for (int i1 = 0; i1 < d; i1++) {
            const float *vi = V + ((long)i1 * 256 + xi[i1]) * K;
            for (int i2 = i1 + 1; i2 < d; i2++) {
                const float *vj = V + ((long)i2 * 256 + xi[i2]) * K;
                for (int l = 0; l < K; l++) score += vi[l] * vj[l];
            }
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```