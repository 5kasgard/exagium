```hypothesis
Family: label-fm (Factorization Machine with 8 latent factors). Adult income prediction benefits from feature interactions (e.g., education×occupation, marital-status×sex) which a linear model misses. A rank-8 FM adds only ~14×256×8 ≈ 28K parameters and ~100 FLOPs/item, keeping inference fast. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower cost. Failure would show interactions aren't learnable with this data/regularization or that FM optimization is unstable.
```
```python
"""Factorization Machine family for tabular label tasks: linear + pairwise interactions via low-rank embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Encoding per card types: ordinal z-scored, categorical one-hot. Linear logits (C=2) plus FM scalar added to class 1, subtracted from class 0.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). K=8 factors.
Exported as per-byte lookup tables: int32 magic 0x464D4C31, d, K, C; float32 Lin[d][256][C], FM[d][256][K], b1[C]."""
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
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoders and Z matrix
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # 256 x 1
            enc.append(E)
            dims.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0                # 256 x num_cats
            enc.append(E)
            dims.append(len(vals))

    # Concatenate to form Z (n x D)
    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    n, D = Z.shape

    # Parameters: W_lin (D x C), b1 (C), V (D x K)
    W_lin = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (D, K)) * math.sqrt(6.0 / (D + K))).astype(np.float32)

    params = [W_lin, b1, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = len(yy)

            # Forward
            lin_logits = z @ W_lin + b1                     # m x C
            S = z @ V                                       # m x K
            S_sq = (z * z) @ (V * V)                        # m x K
            fm = 0.5 * np.sum(S * S - S_sq, axis=1)         # m
            logits = lin_logits.copy()
            logits[:, 0] -= fm
            logits[:, 1] += fm

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_logits = (probs / m).astype(np.float32)    # m x C

            # Gradients for linear part
            g_W_lin = z.T @ grad_logits + L2 * W_lin
            g_b1 = grad_logits.sum(axis=0)

            # Gradient for FM part
            g_fm = grad_logits[:, 1] - grad_logits[:, 0]    # m
            g_S = g_fm[:, None] * S                         # m x K
            g_S_sq = -0.5 * g_fm[:, None]                   # m x K
            g_V = z.T @ g_S + (z * z).T @ (g_S_sq * 2 * V) + L2 * V

            grads = [g_W_lin, g_b1, g_V]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                m_hat = m1 / (1 - 0.9 ** t)
                v_hat = v1 / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-byte lookup tables for export
    Lin_tables = np.zeros((X.shape[1], 256, C), dtype=np.float32)
    FM_tables = np.zeros((X.shape[1], 256, K), dtype=np.float32)
    off = 0
    for j, dim in enumerate(dims):
        Lin_tables[j] = enc[j] @ W_lin[off:off + dim]
        FM_tables[j] = enc[j] @ V[off:off + dim]
        off += dim

    # Pack weights: magic, d, K, C; Lin[d][256][C]; FM[d][256][K]; b1[C]
    magic = 0x464D4C31  # "FML1"
    d = X.shape[1]
    header = struct.pack("<4i", magic, d, K, C)
    out = header + Lin_tables.astype("<f4").tobytes() + FM_tables.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic(0x464D4C31), d, K, C; float32 Lin[d][256][C], FM[d][256][K], b1[C]. */
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
    if (C != 2 || K < 1 || K > 256 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * (C + K) + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *Lin = (const float *)(hd + 4);
    const float *FM = Lin + (long)d * 256 * C;
    const float *b1 = FM + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc_lin = malloc(sizeof(float) * C);
    float *acc_fm = malloc(sizeof(float) * K);
    float *acc_fm_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        acc_lin[0] = b1[0]; acc_lin[1] = b1[1];
        for (int kk = 0; kk < K; kk++) { acc_fm[kk] = 0.0f; acc_fm_sq[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            const float *lin_ptr = Lin + ((long)j * 256 + xi[j]) * C;
            acc_lin[0] += lin_ptr[0];
            acc_lin[1] += lin_ptr[1];

            const float *fm_ptr = FM + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = fm_ptr[kk];
                acc_fm[kk] += v;
                acc_fm_sq[kk] += v * v;
            }
        }

        float fm = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = acc_fm[kk];
            fm += s * s - acc_fm_sq[kk];
        }
        fm *= 0.5f;

        acc_lin[0] -= fm;
        acc_lin[1] += fm;

        unsigned char best = (acc_lin[1] > acc_lin[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```