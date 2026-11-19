```hypothesis
Family: label-fm (Factorization Machine). The 8×8 digit pixels have strong local pairwise interactions (strokes, loops) that a linear model misses. A factorization machine captures all pairwise interactions with only O(d·k) extra parameters (k=8 latent factors) by factorizing the interaction matrix. This should beat label-linear (0.9613) on accuracy while keeping inference cost low (≈2× linear). If dev accuracy ≤ 0.9613 or instructions/item > 10000, the hypothesis fails.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Model: score_c = w0_c + sum_j w_jc * x_j + 0.5 * sum_f [(sum_j V_jfc * x_j)^2 - sum_j (V_jfc^2 * x_j^2)]
Encoding: ordinal features z-scored per byte position. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian): magic(0x464D0001), d, k, C; float32 mean[d], std[d], w0[C], w[d][C], V[d][k][C]."""
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
    rng = np.random.default_rng(SEED)

    # Encode ordinal features: z-score per column
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)  # (n, d)

    n = Z.shape[0]
    # Parameters
    w0 = np.zeros(C, np.float32)
    w = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, y_idx):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return (probs / len(y_idx)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            linear = z @ w + w0  # (B, C)
            inter = np.zeros((len(yy), C), np.float32)
            for f in range(K):
                Vf = V[:, f, :]  # (d, C)
                zv = z @ Vf  # (B, C)
                zv2 = (z * z) @ (Vf * Vf)  # (B, C)
                inter += 0.5 * (zv * zv - zv2)
            logits = linear + inter
            g = softmax_grad(logits, yy)  # (B, C)
            # Gradients
            gw0 = g.sum(axis=0)
            gw = z.T @ g + L2 * w
            gV = np.zeros_like(V)
            for f in range(K):
                Vf = V[:, f, :]
                zv = z @ Vf
                gv = g * zv[:, None, :]  # (B, 1, C) * (B, C) -> (B, C) but need (d, C)
                # gv_jc = sum_b g_bc * z_bj * V_jfc
                # Actually: d/dV_jfc 0.5*(sum_j V_jfc z_j)^2 = V_jfc z_j * sum_l V_lfc z_l = V_jfc z_j * zv_c
                # So grad = g_c * V_jfc * z_j * zv_c - g_c * V_jfc * z_j^2 * V_jfc = g_c * V_jfc * z_j * (zv_c - V_jfc z_j)
                # Better: inter_c = 0.5 * (zv_c^2 - sum_j V_jfc^2 z_j^2)
                # d inter_c / d V_jfc = zv_c * z_j - V_jfc * z_j^2
                # So grad_V_jfc = sum_b g_bc * z_bj * (zv_bc - V_jfc * z_bj)
                zv_exp = zv[:, None, :]  # (B, 1, C)
                z_exp = z[:, :, None]  # (B, d, 1)
                V_exp = Vf[None, :, :]  # (1, d, C)
                g_exp = g[:, None, :]  # (B, 1, C)
                gV[:, f, :] = (g_exp * z_exp * (zv_exp - V_exp * z_exp)).sum(axis=0) + L2 * Vf
            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + C + (long)d * C + (long)d * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *w0   = std + d;
    const float *w    = w0 + C;
    const float *V    = w + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *zv = malloc(sizeof(float) * K * C);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Encode
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];
        // Linear part
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            const float *wj = w + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += zj * wj[c];
        }
        // Interaction part
        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) zv[f * C + c] = 0.0f;
            for (int j = 0; j < d; j++) {
                float zj = z[j];
                const float *Vjfc = V + ((long)j * K + f) * C;
                for (int c = 0; c < C; c++) zv[f * C + c] += zj * Vjfc[c];
            }
        }
        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) {
                float sum = zv[f * C + c];
                float sum_sq = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = V[((long)j * K + f) * C + c];
                    float zj = z[j];
                    sum_sq += v * v * zj * zj;
                }
                scores[c] += 0.5f * (sum * sum - sum_sq);
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```