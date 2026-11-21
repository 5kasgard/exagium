```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which is ideal for the Adult dataset's mix of ordinal and categorical features. They are more expressive than linear models but retain O(d*k) inference cost (d=14 fields, k=embedding size). We expect FM to beat label-linear's 0.8524 accuracy with a modest increase in CPU cost. If dev accuracy does not exceed label-linear by at least 0.5%, the approach fails.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. FM with logistic loss, Adam optimizer.
Exported as: magic 0x464D4C31, d, k, C, global_bias, linear[d*256], embed[d*256*k] (all float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 20
K = 8  # embedding size, chosen on tuning split


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
    assert C == 2

    rng = np.random.default_rng(SEED)

    # Parameters: global bias, linear weights (d, 256), embeddings (d, 256, K)
    b = np.float32(0.0)
    W = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mb, mW, mV = 0.0, np.zeros_like(W), np.zeros_like(V)
    vb, vW, vV = 0.0, np.zeros_like(W), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            linear = b + np.sum(W[np.arange(d), xb.T], axis=0)  # (B,)
            # Embeddings sum
            V_sum = np.zeros((B, K), dtype=np.float32)
            V_sq_sum = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                vj = V[j, xb[:, j]]  # (B, K)
                V_sum += vj
                V_sq_sum += vj * vj
            inter = 0.5 * np.sum(V_sum * V_sum - V_sq_sum, axis=1)
            score = linear + inter  # (B,)

            # Logistic loss gradient w.r.t. score
            # labels: 0 -> -1, 1 -> +1
            y_sign = 2 * yb - 1
            z = y_sign * score
            sig = 1.0 / (1.0 + np.exp(-z))
            grad_score = (sig - 1.0) * y_sign  # (B,)

            # Gradients
            gb = np.mean(grad_score)
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)

            for j in range(d):
                vals = xb[:, j]
                # linear gradient
                np.add.at(gW[j], vals, grad_score)
                gW[j] /= B
                # embedding gradient
                # For each example, d(inter)/dV_j = (V_sum - vj) * grad_score
                vj = V[j, vals]  # (B, K)
                factor = (V_sum - vj) * grad_score[:, None]  # (B, K)
                np.add.at(gV[j], vals, factor)
                gV[j] /= B

            # L2 regularization
            gW += L2 * W
            gV += L2 * V

            t += 1
            # Adam update
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * (gV * gV)

            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)

            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + eps)

    # Pack weights
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += np.array([b], dtype=np.float32).tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index (0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 16);
    float global_bias = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *V_sum = malloc(sizeof(float) * K);
    float *V_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = global_bias;
        for (int j = 0; j < K; j++) V_sum[j] = 0.0f, V_sq_sum[j] = 0.0f;

        for (int j = 0; j < d; j++) {
            int val = xi[j];
            score += W[(long)j * 256 + val];
            const float *vj = V + ((long)j * 256 + val) * K;
            for (int u = 0; u < K; u++) {
                V_sum[u] += vj[u];
                V_sq_sum[u] += vj[u] * vj[u];
            }
        }
        for (int u = 0; u < K; u++) {
            score += 0.5f * (V_sum[u] * V_sum[u] - V_sq_sum[u]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```