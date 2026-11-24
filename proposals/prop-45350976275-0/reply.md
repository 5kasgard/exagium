```hypothesis
Factorization Machine (label-fm) with rank-8 per-class latent factors captures categorical feature interactions (education×occupation, marital-status×sex) that linear models miss. The Adult dataset's 8 categorical columns with moderate cardinality are ideal for FM. Precomputed per-column lookup tables yield O(d·C·k) inference (≈224 ops/item), ~3× label-linear cost but targeting 0.86+ accuracy. Failure: dev accuracy ≤0.855 or instructions/item >3000.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot over training values.
FM: per-class linear weights + per-class rank-K factors.
Adam lr 1e-2, L2 1e-4, minibatch 64, epochs min(100, ceil(600000 / n)).
Exported as per-byte lookup tables: int32 magic 0x464D31, d, K, C;
float32 T_lin[d][256][C], T_fm[d][256][C][K], T_corr[d][256][C][K], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    d_in = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Build encoders per column, track expanded dims and ordinal mask
    enc = []
    enc_dim = []
    is_ordinal_expanded = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # 256 x 1
            enc.append(E)
            enc_dim.append(1)
            is_ordinal_expanded.append(True)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            enc_dim.append(len(vals))
            is_ordinal_expanded.append(False)

    # Expanded feature matrix Z (n x d_exp)
    d_exp = sum(enc_dim)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Ordinal mask in expanded space
    ordinal_mask = np.zeros(d_exp, dtype=bool)
    off = 0
    for j, dim in enumerate(enc_dim):
        if is_ordinal_expanded[j]:
            ordinal_mask[off] = True
        off += dim

    # Z_corr: for ordinal features, square the value; categorical unchanged (0/1)
    Z_corr = Z.copy()
    Z_corr[:, ordinal_mask] = Z[:, ordinal_mask] ** 2

    # Parameters
    W = (rng.uniform(-1, 1, (d_exp, C)) * math.sqrt(6.0 / d_exp)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d_exp, K, C)) * math.sqrt(6.0 / (d_exp * K))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [W, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, zc, yy = Z[idx], Z_corr[idx], y[idx]
            m = len(yy)

            # Forward
            lin = z @ W + bias  # m x C
            fm = np.zeros((m, C), np.float32)
            S = np.zeros((m, K, C), np.float32)
            Cc = np.zeros((m, K, C), np.float32)
            for c in range(C):
                for f in range(K):
                    Vf = V[:, f, c]
                    Sf = z @ Vf
                    Ccf = zc @ (Vf * Vf)
                    S[:, f, c] = Sf
                    Cc[:, f, c] = Ccf
                    fm[:, c] += 0.5 * (Sf * Sf - Ccf)
            scores = lin + fm

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_scores = probs / m  # m x C

            # Gradients
            grad_W = z.T @ grad_scores + L2 * W
            grad_bias = grad_scores.sum(axis=0)

            grad_V = np.zeros_like(V)
            for c in range(C):
                Gc = grad_scores[:, c]  # m
                for f in range(K):
                    Vf = V[:, f, c]
                    Sf = S[:, f, c]
                    # dL/dV = Z.T @ (Gc * Sf) - (Z_corr.T @ Gc) * Vf + L2 * Vf
                    grad_V[:, f, c] = z.T @ (Gc * Sf) - (zc.T @ Gc) * Vf + L2 * Vf

            grads = [grad_W, grad_V, grad_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m_ / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-column lookup tables
    # T_lin[j][v][c], T_fm[j][v][c][f], T_corr[j][v][c][f]
    T_lin = np.zeros((d_in, 256, C), np.float32)
    T_fm = np.zeros((d_in, 256, C, K), np.float32)
    T_corr = np.zeros((d_in, 256, C, K), np.float32)

    off = 0
    for j in range(d_in):
        dim = enc_dim[j]
        E = enc[j]  # 256 x dim
        Wj = W[off:off + dim]  # dim x C
        Vj = V[off:off + dim]  # dim x K x C
        ordinal = is_ordinal_expanded[j]

        if ordinal:
            # E is 256 x 1, z-scored values
            z_vals = E[:, 0]  # 256
            z2_vals = z_vals * z_vals
            for c in range(C):
                T_lin[j, :, c] = z_vals * Wj[0, c]
                for f in range(K):
                    Vf = Vj[0, f, c]
                    T_fm[j, :, c, f] = z_vals * Vf
                    T_corr[j, :, c, f] = z2_vals * (Vf * Vf)
        else:
            # E is 256 x dim, one-hot rows
            for c in range(C):
                T_lin[j, :, c] = E @ Wj[:, c]
                for f in range(K):
                    T_fm[j, :, c, f] = E @ Vj[:, f, c]
                    T_corr[j, :, c, f] = E @ (Vj[:, f, c] ** 2)
        off += dim

    # Pack weights
    # Magic, d_in, K, C
    out = struct.pack("<4i", 0x464D31, d_in, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_fm.astype("<f4").tobytes()
    out += T_corr.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (rank-K) for label tasks: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    /* Layout: T_lin[d][256][C], T_fm[d][256][C][K], T_corr[d][256][C][K], bias[C] */
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * C * K
        + 4L * d * 256 * C * K
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *T_lin = p; p += d * 256 * C;
    const float *T_fm = p; p += d * 256 * C * K;
    const float *T_corr = p; p += d * 256 * C * K;
    const float *bias = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score[256];
        for (int c = 0; c < C; c++) score[c] = bias[c];

        float S[256][64];
        float Csum[256][64];
        for (int c = 0; c < C; c++)
            for (int f = 0; f < K; f++)
                S[c][f] = 0.0f, Csum[c][f] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin = T_lin + ((long)j * 256 + v) * C;
            const float *fm = T_fm + ((long)j * 256 + v) * C * K;
            const float *corr = T_corr + ((long)j * 256 + v) * C * K;
            for (int c = 0; c < C; c++) {
                score[c] += lin[c];
                for (int f = 0; f < K; f++) {
                    S[c][f] += fm[c * K + f];
                    Csum[c][f] += corr[c * K + f];
                }
            }
        }

        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) {
                float s = S[c][f];
                score[c] += 0.5f * (s * s - Csum[c][f]);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```