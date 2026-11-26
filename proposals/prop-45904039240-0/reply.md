```hypothesis
Family: label-fm (factorization machine with 8-dimensional embeddings).
Why: Adult income prediction benefits from feature interactions (e.g., education × occupation). A factorization machine models all pairwise interactions with O(d*k) parameters, capturing non-linearities while staying compact. With k=8 it should surpass label-linear (0.8524) on dev accuracy.
Failure: If dev accuracy ≤ 0.8524, the hypothesis is falsified.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Learns per-feature per-value biases and k-dimensional embeddings.
Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias[d][256], emb[d][256][k], global_bias."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM implementation assumes binary classification"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias[d,256], emb[d,256,K], global_bias
    bias = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    global_bias = np.float32(0.0)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_gb, v_gb = np.float32(0.0), np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (bs, d)
            yb = y[idx]          # (bs,)
            bs = len(yb)

            # Gather parameters for this batch
            w_batch = bias[:, xb.T]                 # (d, bs)
            e_batch = emb[:, xb.T, :]               # (d, bs, K)

            # Forward
            linear = global_bias + np.sum(w_batch, axis=0)                     # (bs,)
            sum_e = np.sum(e_batch, axis=0)                                    # (bs, K)
            sum_sq_e = np.sum(e_batch * e_batch, axis=0)                       # (bs, K)
            interaction = 0.5 * np.sum(sum_e * sum_e - sum_sq_e, axis=1)       # (bs,)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            # Gradient of loss w.r.t logit
            dlogit = (p - yb) / bs                                             # (bs,)

            # Gradients
            db = np.sum(dlogit)
            dw = np.zeros_like(bias)
            de = np.zeros_like(emb)

            # Bias gradients: add dlogit to each feature's value
            for j in range(d):
                np.add.at(dw[j], xb[:, j], dlogit)

            # Embedding gradients
            for j in range(d):
                vals = xb[:, j]
                # dlogit * (sum_e - e_batch[:,j,:])
                de_interaction = dlogit[:, None] * (sum_e - e_batch[j])        # (bs, K)
                np.add.at(de[j], vals, de_interaction)

            # L2 regularization
            dw += L2 * bias
            de += L2 * emb
            db += L2 * global_bias

            # Adam updates
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * dw
            v_bias = 0.999 * v_bias + 0.001 * dw * dw
            m_bias_hat = m_bias / (1 - 0.9 ** t)
            v_bias_hat = v_bias / (1 - 0.999 ** t)
            bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * de
            v_emb = 0.999 * v_emb + 0.001 * de * de
            m_emb_hat = m_emb / (1 - 0.9 ** t)
            v_emb_hat = v_emb / (1 - 0.999 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + 1e-8)
            # global_bias
            m_gb = 0.9 * m_gb + 0.1 * db
            v_gb = 0.999 * v_gb + 0.001 * db * db
            m_gb_hat = m_gb / (1 - 0.9 ** t)
            v_gb_hat = v_gb / (1 - 0.999 ** t)
            global_bias -= LR * m_gb_hat / (np.sqrt(v_gb_hat) + 1e-8)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    # bias: d*256 floats
    # emb: d*256*K floats
    # global_bias: 1 float
    out = header + bias.astype("<f4").tobytes() + emb.astype("<f4").tobytes() + np.array([global_bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)d * 256 * K + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *emb = bias + (long)d * 256;
    const float *global_bias = emb + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = (float *)malloc(sizeof(float) * K);
    float *sum_sq_e = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = global_bias[0];
        for (int f = 0; f < K; f++) { sum_e[f] = 0.0f; sum_sq_e[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += bias[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_e[f] += val;
                sum_sq_e[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_e[f] * sum_e[f] - sum_sq_e[f]);
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_e); free(sum_sq_e);
    return 0;
}
```