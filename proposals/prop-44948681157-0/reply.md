```hypothesis
Factorization Machines (FM) of rank 2 capture pairwise feature interactions efficiently, which are known to be effective on tabular data like Adult. With 14 features and binary classification, a rank-2 FM adds ~14K parameters (vs 7K for linear) and minimal inference overhead (O(d·k·C) operations). We expect dev accuracy > 0.8524 (label-linear) with similar latency. If dev accuracy does not exceed linear, it shows pairwise interactions don't help on this dataset.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Exports: int32 magic 0x464D4C31, d, k, C; float32 linear[d][256][C], fm[d][256][k][C], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 2                # FM rank
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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: linear[d, 256, C], fm[d, 256, K, C]
    linear = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / C)).astype(np.float32)
    fm = (rng.uniform(-1, 1, (d, 256, K, C)) * math.sqrt(6.0 / (K * C))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    params = [linear, fm, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]

            # Forward: logits = bias + sum_j linear[j, x_j] + 0.5 * sum_f ( (sum_j fm[j, x_j, f])^2 - sum_j fm[j, x_j, f]^2 )
            B = xb.shape[0]
            logits = np.tile(bias, (B, 1))                 # [B, C]
            # linear part
            for j in range(d):
                logits += linear[j, xb[:, j]]               # [B, C]
            # FM part
            for f in range(K):
                sum_f = np.zeros((B, C), dtype=np.float32)
                sum_sq_f = np.zeros((B, C), dtype=np.float32)
                for j in range(d):
                    v = fm[j, xb[:, j], f]                  # [B, C]
                    sum_f += v
                    sum_sq_f += v * v
                logits += 0.5 * (sum_f * sum_f - sum_sq_f)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)    # [B, C]

            # Backward
            grad_linear = np.zeros_like(linear)
            grad_fm = np.zeros_like(fm)
            grad_bias = grad_logits.sum(axis=0)

            # linear gradients
            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], grad_logits)

            # FM gradients
            for f in range(K):
                sum_f = np.zeros((B, C), dtype=np.float32)
                for j in range(d):
                    sum_f += fm[j, xb[:, j], f]
                # derivative of 0.5*(sum^2 - sum_sq) w.r.t v = sum_f - v
                for j in range(d):
                    v = fm[j, xb[:, j], f]                  # [B, C]
                    g = grad_logits * (sum_f - v)           # [B, C]
                    np.add.at(grad_fm[j, :, f], xb[:, j], g)

            grads = [grad_linear + L2 * linear, grad_fm + L2 * fm, grad_bias + L2 * bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += fm.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * K * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *fm = linear + (long)d * 256 * C;
    const float *bias = fm + (long)d * 256 * K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logits[256];
        for (int c = 0; c < C; c++) logits[c] = bias[c];

        // linear part
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        // FM part
        for (int f = 0; f < K; f++) {
            float sum_f[256] = {0};
            float sum_sq_f[256] = {0};
            for (int j = 0; j < d; j++) {
                const float *v = fm + ((long)j * 256 + xi[j]) * K * C + f * C;
                for (int c = 0; c < C; c++) {
                    float val = v[c];
                    sum_f[c] += val;
                    sum_sq_f[c] += val * val;
                }
            }
            for (int c = 0; c < C; c++) {
                logits[c] += 0.5f * (sum_f[c] * sum_f[c] - sum_sq_f[c]);
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```