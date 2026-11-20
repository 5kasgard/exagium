```hypothesis
Family: label-fm2 (Factorization Machine rank-4 + linear). The Adult dataset has known feature interactions (e.g., education × occupation, marital-status × relationship) that linear models miss. A rank-4 FM adds pairwise interactions via low-rank embeddings (14×256×4 ≈ 14K params) while keeping inference at ~3× linear cost (≈2200 Ir/item) by computing 0.5·(‖Σv‖²−Σ‖v‖²) per example. If dev accuracy ≤ 0.855 or Ir/item ≥ 3000, the hypothesis fails.
```
```python
"""Factorization Machine (rank-4) + linear for binary tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D324C, d, r, C; float32 W_lin[d][256][C], W_fm[d][256][r], bias[C]."""
import json, math, struct, sys

import numpy as np

R = 4                # FM rank
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
    d = card["input"]["bytes"]
    C = card["classes"]          # 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: W_lin[d,256,C], W_fm[d,256,R], bias[C]
    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    W_fm  = rng.normal(0, 0.01, (d, 256, R)).astype(np.float32)
    bias  = np.zeros(C, np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_fm,  v_fm  = np.zeros_like(W_fm),  np.zeros_like(W_fm)
    m_b,   v_b   = np.zeros_like(bias),  np.zeros_like(bias)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = X[batch_idx]
            yb = y[batch_idx]
            bsz = len(yb)

            # Forward
            # Linear scores
            lin_scores = bias + np.sum(W_lin[np.arange(d)[:,None], xb.T, :], axis=0)  # (bsz, C)
            # FM scores (only added to class 1)
            S = np.zeros((bsz, R), np.float32)
            Q = np.zeros((bsz, R), np.float32)
            for j in range(d):
                v = W_fm[j, xb[:, j], :]          # (bsz, R)
                S += v
                Q += v * v
            fm = 0.5 * np.sum(S * S - Q, axis=1)  # (bsz,)
            scores = lin_scores
            scores[:, 1] += fm

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad = probs / bsz                       # (bsz, C)

            # Gradients
            # Linear
            g_lin = np.zeros_like(W_lin)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad)
            g_lin += L2 * W_lin

            # FM: only class-1 gradient flows into FM
            g1 = grad[:, 1]                          # (bsz,)
            g_fm = np.zeros_like(W_fm)
            for j in range(d):
                vals = xb[:, j]
                v = W_fm[j, vals, :]                 # (bsz, R)
                # dfm/dv = S_k - v_jk
                gv = g1[:, None] * (S - v)           # (bsz, R)
                np.add.at(g_fm[j], vals, gv)
            g_fm += L2 * W_fm

            g_bias = grad.sum(axis=0) + L2 * bias

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for param, grad_p, m, v in [
                (W_lin, g_lin, m_lin, v_lin),
                (W_fm,  g_fm,  m_fm,  v_fm),
                (bias,  g_bias, m_b,   v_b)
            ]:
                m *= beta1; m += (1 - beta1) * grad_p
                v *= beta2; v += (1 - beta2) * grad_p * grad_p
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    magic = 0x464D324C  # "FM2L"
    out = struct.pack("<4i", magic, d, R, C)
    out += W_lin.astype("<f4").tobytes()
    out += W_fm.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (rank-4) + linear for binary classification.
 * ./infer weights.bin in.rec N
 * Weights: magic 0x464D324C, d, R, C; float32 W_lin[d][256][C], W_fm[d][256][R], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D324C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C != 2 || R != 4 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * R + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *W_fm  = p; p += (long)d * 256 * R;
    const float *bias  = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float score[2], S[4], Q[4];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        score[0] = bias[0]; score[1] = bias[1];
        for (int r = 0; r < R; r++) { S[r] = 0.0f; Q[r] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin = W_lin + ((long)j * 256 + v) * C;
            score[0] += lin[0];
            score[1] += lin[1];
            const float *fm = W_fm + ((long)j * 256 + v) * R;
            for (int r = 0; r < R; r++) {
                float fv = fm[r];
                S[r] += fv;
                Q[r] += fv * fv;
            }
        }
        float fm_score = 0.0f;
        for (int r = 0; r < R; r++) fm_score += S[r] * S[r] - Q[r];
        score[1] += 0.5f * fm_score;

        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```