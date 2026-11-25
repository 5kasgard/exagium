```hypothesis
Family: label-fm (Factorization Machine with shared embeddings, k=4).
Why: Adult has 14 mixed-type features already quantized to 256 bins. A degree-2 FM captures pairwise interactions with O(d·k) parameters and inference cost, far fewer than an MLP. Shared embeddings across the two classes keep the model tiny (≈21k floats) and inference fast (≈14·k FM ops + linear). Beats linear on accuracy, matches or beats MLP speed.
Failure signal: dev accuracy ≤ label-linear (0.8524) or Ir/item > label-mlp16 (2325).
```
```python
"""Factorization Machine (degree 2, shared embeddings) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, int32 d, k, C; float32 linear[C][d][256], emb[d][256][k], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 4                     # embedding dimension (tuned on last 20% of train.in)


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
    k = K
    rng = np.random.default_rng(SEED)

    # parameters
    linear = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    emb    = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)
    bias   = np.zeros(C, np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb),    np.zeros_like(emb)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                     # (B, d)
            yb = y[idx]                     # (B,)
            B = xb.shape[0]

            # forward
            lin_logits = bias + np.sum(linear[:, np.arange(d), xb.T], axis=1).T      # (B, C)
            e = emb[np.arange(d), xb.T, :]                                           # (d, B, k)
            e = np.transpose(e, (1, 0, 2))                                           # (B, d, k)
            emb_sum = e.sum(axis=1)                                                  # (B, k)
            emb_sq_sum = (e * e).sum(axis=1)                                         # (B, k)
            inter = 0.5 * ((emb_sum * emb_sum - emb_sq_sum).sum(axis=1))             # (B,)
            logits = lin_logits + inter[:, None]                                     # (B, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                                                  # (B, C)

            # gradients
            g_bias = grad_logits.sum(axis=0)                                         # (C,)
            g_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_linear, (slice(None), j, xb[:, j]), grad_logits)

            g_inter = grad_logits.sum(axis=1)                                        # (B,)
            diff = emb_sum[:, None, :] - e                                           # (B, d, k)
            weighted_diff = g_inter[:, None, None] * diff                            # (B, d, k)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb, (j, xb[:, j], slice(None)), weighted_diff[:, j, :])

            # L2 regularization
            g_linear += L2 * linear
            g_emb    += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (linear, g_linear, m_lin, v_lin),
                (emb,    g_emb,    m_emb, v_emb),
                (bias,   g_bias,   m_bias, v_bias)
            ]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, k, C)
    body = (
        linear.astype("<f4").tobytes() +
        emb.astype("<f4").tobytes() +
        bias.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2, shared embeddings) inference.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d from header). Output: 1 byte class index. */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || d < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected_floats = (long)C * d * 256 + (long)d * 256 * k + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (long)C * d * 256;
    const float *bias = emb + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * k);
    float *emb_sq_sum = malloc(sizeof(float) * k);
    float logits[2];  // C <= 2 for this task

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // linear part
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++)
                logits[c] += linear[((long)c * d + j) * 256 + v];
        }

        // FM interaction (shared)
        memset(emb_sum, 0, sizeof(float) * k);
        memset(emb_sq_sum, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                emb_sum[f] += val;
                emb_sq_sum[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += emb_sum[f] * emb_sum[f] - emb_sq_sum[f];
        inter *= 0.5f;

        for (int c = 0; c < C; c++) logits[c] += inter;

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(emb_sq_sum);
    return 0;
}
```