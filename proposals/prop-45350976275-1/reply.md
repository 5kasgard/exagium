```hypothesis
The label-fm family uses a Factorization Machine with shared latent factors (k=8) to model pairwise pixel interactions, followed by a per-class linear readout. On optdigits, digit shapes depend on stroke/corner interactions (e.g., loops in 8, crossbars in 4,5); FM captures these with only ~1200 params and ~1152 table lookups/item (vs label-linear's 640). It should exceed 96.1% dev accuracy at ~1.5x CPU cost. Failure: accuracy ≤ 96.1% or Ir/item > 10000.
```
```python
"""Factorization Machine (shared factors, per-class readout) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored per byte. Model: linear + 0.5*sum_f((sum_j V_jf z_j)^2 - sum_j (V_jf z_j)^2) per class.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1.
Exported tables: int32 magic 0x464D0001, d, k, C; float32 lin[d][256][C], fac[d][256][k], bias[C], U[C][k]."""
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices E_j[v] = (v - mu_j)/sd_j for all 256 byte values
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd).astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E[:, 0])  # fallback, but all are ordinal here

    # Normalized training features Z[n, d]
    Z = np.column_stack([enc[j][X[:, j]] for j in range(d)]).astype(np.float32)
    n = Z.shape[0]

    # Parameters
    W_lin = (rng.normal(0, 0.01, (d, C)) * math.sqrt(2.0 / d)).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, K)) * math.sqrt(2.0 / d)).astype(np.float32)
    U = (rng.normal(0, 0.01, (C, K)) * math.sqrt(2.0 / K)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [W_lin, V, U, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            lin = z @ W_lin + bias                          # (B, C)
            s1 = z @ V                                      # (B, K)
            s2 = (z * z) @ (V * V)                          # (B, K)
            fm = 0.5 * (s1 * s1 - s2)                       # (B, K)
            logits = lin + fm @ U.T                         # (B, C)
            # Backward
            g = softmax_grad(logits, yy)                    # (B, C)
            # grad bias
            g_bias = g.sum(axis=0)
            # grad W_lin
            g_Wlin = z.T @ g + L2 * W_lin
            # grad U
            g_U = g.T @ fm + L2 * U
            # grad V
            # d logits / d fm = g @ U  (B, K)
            g_fm = g @ U                                    # (B, K)
            # d fm / d s1 = s1, d fm / d s2 = -0.5
            g_s1 = g_fm * s1                                # (B, K)
            g_s2 = g_fm * -0.5                              # (B, K)
            # d s1 / d V = z^T, d s2 / d V = 2 * z^2 * V (but s2 = sum_j z_j^2 V_jf^2)
            # Actually s2_f = sum_j z_j^2 V_jf^2, so ds2_f/dV_jf = 2 z_j^2 V_jf
            # So grad V = z^T @ g_s1 + (z^2)^T @ (g_s2 * 2 * V)
            g_V = z.T @ g_s1 + (z * z).T @ (g_s2 * 2.0 * V) + L2 * V

            grads = [g_Wlin, g_V, g_U, g_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export tables: fold encodings into tables for all 256 byte values
    # lin_table[j, v, c] = W_lin[j, c] * enc[j][v]
    # fac_table[j, v, f] = V[j, f] * enc[j][v]
    lin_table = np.zeros((d, 256, C), np.float32)
    fac_table = np.zeros((d, 256, K), np.float32)
    for j in range(d):
        e = enc[j][:, None]                  # (256, 1)
        lin_table[j] = e @ W_lin[j:j+1, :]   # (256, C)
        fac_table[j] = e @ V[j:j+1, :]       # (256, K)

    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += lin_table.astype("<f4").tobytes()
    out += fac_table.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (shared factors, per-class readout) for label tasks.
 * ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + C + (long)C * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *fac = lin + (long)d * 256 * C;
    const float *bias = fac + (long)d * 256 * K;
    const float *U = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *s1 = malloc(sizeof(float) * K);
    float *s2 = malloc(sizeof(float) * K);
    float *fm = malloc(sizeof(float) * K);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) lin_acc[c] = bias[c];
        for (int f = 0; f < K; f++) { s1[f] = 0.0f; s2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_row = lin + ((long)j * 256 + v) * C;
            const float *fac_row = fac + ((long)j * 256 + v) * K;
            for (int c = 0; c < C; c++) lin_acc[c] += lin_row[c];
            for (int f = 0; f < K; f++) {
                float val = fac_row[f];
                s1[f] += val;
                s2[f] += val * val;
            }
        }

        for (int f = 0; f < K; f++) fm[f] = 0.5f * (s1[f] * s1[f] - s2[f]);

        for (int c = 0; c < C; c++) {
            float sc = lin_acc[c];
            const float *U_row = U + (long)c * K;
            for (int f = 0; f < K; f++) sc += U_row[f] * fm[f];
            score[c] = sc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```