```hypothesis
Family: label-fm (Factorization Machine). Adult has strong categorical interactions (education×occupation, marital×relationship) that linear models miss. FM captures all pairwise interactions via low-rank latent vectors (k=8) with only ~50k params. Expect dev accuracy ≥0.860 (beating label-linear 0.852, near label-mlp16 0.859) at ~3× linear inference cost. Failure: accuracy ≤0.853 or latency >4× linear.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. FM with per-class latent vectors (k=8).
Fields = input bytes (d=14), each 256 values. Linear + pairwise interactions via k factors.
Adam lr=0.01, L2_lin=1e-4, L2_lat=1e-4, batch=256, epochs=min(50, ceil(200000/n)).
Exported: magic 0x464D3132, d, k, C; float32 T_lin[d][256][C], T_lat[d][256][C][k], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_LIN, L2_LAT, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 1e-4, 256, 200000, 50
MAGIC = 0x464D3132  # "FM12"


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
    assert d == 14 and C == 2

    rng = np.random.default_rng(SEED)
    W_lin = np.zeros((d, 256, C), np.float32)
    W_lat = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [W_lin, W_lat, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward
            lin_scores = bias.copy()
            for j in range(d):
                lin_scores += W_lin[j, xb[:, j], :]

            sum_v = np.zeros((B, C, K), np.float32)
            sum_v2 = np.zeros((B, C, K), np.float32)
            for j in range(d):
                w = W_lat[j, xb[:, j], :, :]
                sum_v += w
                sum_v2 += w * w

            inter_scores = 0.5 * (sum_v * sum_v - sum_v2).sum(axis=2)
            scores = lin_scores + inter_scores

            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dS = probs / B  # (B, C)

            # Gradients
            grad_lin = np.zeros_like(W_lin)
            grad_lat = np.zeros_like(W_lat)
            for j in range(d):
                # Linear: scatter-add dS (B, C) into grad_lin[j] (256, C)
                np.add.at(grad_lin[j], xb[:, j], dS)
                # Latent: scatter-add dS[:,:,None] * (sum_v - W_lat[j, xb[:,j], :, :]) (B, C, K)
                w = W_lat[j, xb[:, j], :, :]
                diff = sum_v - w
                updates = (dS[:, :, None] * diff).reshape(B, -1)
                np.add.at(grad_lat[j].reshape(256, -1), xb[:, j], updates)

            grad_lin += L2_LIN * W_lin
            grad_lat += L2_LAT * W_lat
            grad_bias = dS.sum(axis=0)

            grads = [grad_lin, grad_lat, grad_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: T_lin (d,256,C), T_lat (d,256,C,K), bias (C) — all little-endian float32
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += W_lat.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C * (1 + K) + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_lat = T_lin + (long)d * 256 * C;
    const float *bias = T_lat + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * C * K);
    float *sum_v2 = malloc(sizeof(float) * C * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *lin = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lin[c];
        }

        for (int c = 0; c < C; c++)
            for (int f = 0; f < K; f++) {
                sum_v[c * K + f] = 0.0f;
                sum_v2[c * K + f] = 0.0f;
            }

        for (int j = 0; j < d; j++) {
            const float *lat = T_lat + (((long)j * 256 + xi[j]) * C) * K;
            for