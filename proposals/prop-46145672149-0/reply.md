```hypothesis
Family: label-fm (factorization machine with rank-4 interactions).
Why: Adult has strong pairwise interactions (e.g., education × occupation, marital-status × relationship). A rank-4 FM adds only ~140 KB weights and ~250 adds/muls per item, capturing 2nd-order effects that linear models miss. The low-rank factorization keeps inference fast and weights small.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 1500, the interaction gain does not justify the cost.
```

```python
"""Factorization Machine (rank k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x464D3231, d, k, C; float32 linear_emb[d][256][C], inter_emb[d][256][C][k], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4  # FM rank


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

    rng = np.random.default_rng(SEED)
    # Parameters
    linear_emb = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    inter_emb = rng.normal(0, 0.01, (d, 256, C, K)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [linear_emb, inter_emb, bias]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            linear_score = np.zeros((bsz, C), np.float32)
            inter_sum = np.zeros((bsz, C, K), np.float32)
            inter_sq_sum = np.zeros((bsz, C, K), np.float32)
            for j in range(d):
                vals = xb[:, j]
                linear_score += linear_emb[j, vals]          # (bsz, C)
                ival = inter_emb[j, vals]                    # (bsz, C, K)
                inter_sum += ival
                inter_sq_sum += ival * ival
            interaction = 0.5 * ((inter_sum ** 2 - inter_sq_sum).sum(axis=2))  # (bsz, C)
            scores = linear_score + interaction + bias  # (bsz, C)

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            dscores = (probs / bsz).astype(np.float32)  # (bsz, C)

            # Gradients
            grad_linear = np.zeros_like(linear_emb)
            grad_inter = np.zeros_like(inter_emb)
            grad_bias = dscores.sum(axis=0)

            # d(interaction)/d(inter_emb) = inter_sum - ival
            dinter = dscores[:, :, None] * (inter_sum - inter_emb[j, xb[:, j]])  # WRONG: inter_emb[j] not defined here
            # We need to compute per column. Let's loop over columns for gradients.
            for j in range(d):
                vals = xb[:, j]
                # Linear gradient
                for c in range(C):
                    np.add.at(grad_linear[j], (vals, c), dscores[:, c])
                # Interaction gradient
                ival = inter_emb[j, vals]                 # (bsz, C, K)
                # dinter for this column: dscores * (inter_sum - ival)
                din = dscores[:, :, None] * (inter_sum - ival)  # (bsz, C, K)
                for c in range(C):
                    for f in range(K):
                        np.add.at(grad_inter[j], (vals, c, f), din[:, c, f])

            # L2 regularization
            grad_linear += L2 * linear_emb
            grad_inter += L2 * inter_emb
            grad_bias += L2 * bias

            grads = [grad_linear, grad_inter, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3231  # "FM21"
    total_floats = d * 256 * C * (1 + K) + C
    out = struct.pack("<4i", magic, d, K, C)
    out += linear_emb.astype("<f4").tobytes()
    out += inter_emb.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (rank k) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C * (1 + K) + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear_emb = (const float *)(hd + 4);
    const float *inter_emb = linear_emb + (long)d * 256 * C;
    const float *bias = inter_emb + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score[256];  // C <= 256
        float sum_inter[256][16];  // C x K, K <= 16
        float sum_inter_sq[256][16];
        for (int c = 0; c < C; c++) {
            score[c] = bias[c];
            for (int f = 0; f < K; f++) {
                sum_inter[c][f] = 0.0f;
                sum_inter_sq[c][f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            long base_linear = ((long)j * 256 + val) * C;
            long base_inter = ((long)j * 256 + val) * C * K;
            for (int c = 0; c < C; c++) {
                score[c] += linear_emb[base_linear + c];
                long base_inter_c = base_inter + (long)c * K;
                for (int f = 0; f < K; f++) {
                    float v = inter_emb[base_inter_c + f];
                    sum_inter[c][f] += v;
                    sum_inter_sq[c][f] += v * v;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = sum_inter[c][f];
                inter += s * s - sum_inter_sq[c][f];
            }
            score[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```