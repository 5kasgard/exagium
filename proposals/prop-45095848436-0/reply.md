```hypothesis
Factorization Machine (FM) with latent dimension k=8 captures pairwise feature interactions (e.g., education×occupation, marital-status×sex) crucial for tabular data. Same O(d) encoding as label-linear but adds O(d*k) interaction computation; Adult typically gains 0.5-1% accuracy over logistic regression. Failure: dev accuracy ≤ 0.853 (no meaningful gain) or inference cost > 2× label-linear.
```
```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via low-rank factors.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Encoding: ordinal z-scored, categorical one-hot over training values. FM with latent dim K.
Adam lr 1e-2, L2 1e-4 (linear) / 1e-3 (factors), minibatch 256, epochs min(50, ceil(300000 / n)).
Exported as per-byte lookup tables: magic 0x464D3130, d, K, C; float32 T_lin[d][256], T_fac[d][256][K], b_lin[C], b_fac[K][C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_LIN, L2_FAC, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 1e-3, 256, 300000, 50


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

    # Build encoders per column
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
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            dims.append(len(vals))

    # Total feature dimension after encoding
    D = sum(dims)
    # FM parameters: linear weights D x C, factor matrix D x K
    W = (rng.normal(0, 0.01, (D, C)) * math.sqrt(2.0 / D)).astype(np.float32)
    V = (rng.normal(0, 0.01, (D, K)) * math.sqrt(2.0 / D)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    b_fac = np.zeros((K, C), np.float32)

    # Precompute encoded design matrix Z (n x D)
    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    n = Z.shape[0]

    # Adam state
    params = [W, V, b_lin, b_fac]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # B x D
            yy = y[idx]         # B

            # FM forward: linear + interaction
            # linear: z @ W + b_lin  -> B x C
            # interaction: 0.5 * sum_k ((z @ V[:,k])^2 - z^2 @ V[:,k]^2) + b_fac
            # We need gradients wrt W, V, b_lin, b_fac
            lin_part = z @ W + b_lin          # B x C
            # For each factor k: (z @ V[:,k]) is B-vector
            zV = z @ V                        # B x K
            z2 = z * z
            V2 = V * V
            zV2 = z2 @ V2                     # B x K
            inter_part = 0.5 * (zV * zV - zV2) @ b_fac  # B x C  (b_fac is K x C)
            logits = lin_part + inter_part

            g = softmax_grad(logits, yy)      # B x C

            # Gradients
            grad_W = z.T @ g + L2_LIN * W
            grad_b_lin = g.sum(0)

            # Gradient for V: for each sample, each factor k, each feature d:
            # d(inter)/dV[d,k] = g * b_fac[k] * (zV[:,k] * z[:,d] - z2[:,d] * V[d,k])
            # Vectorized:
            # g_b = g @ b_fac.T   # B x K
            # For each k: g_b[:,k] * (zV[:,k] * z - z2 * V[:,k])
            g_b = g @ b_fac.T                 # B x K
            grad_V = np.zeros_like(V)
            for k in range(K):
                # g_b[:,k] * zV[:,k] -> B, outer with z gives B x D
                # Actually: sum_b g_b[b,k] * zV[b,k] * z[b,:]
                term1 = (g_b[:, k][:, None] * zV[:, k][:, None]) * z
                term2 = (g_b[:, k][:, None] * z2) * V[:, k][None, :]
                grad_V[:, k] = (term1 - term2).sum(0)
            grad_V += L2_FAC * V

            grad_b_fac = 0.5 * (zV * zV - zV2).T @ g  # K x C

            grads = [grad_W, grad_V, grad_b_lin, grad_b_fac]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables
    # T_lin[j][v] = sum over features f in column j of E_j[v,f] * W[f]  -> C-vector
    # T_fac[j][v][k] = sum_f E_j[v,f] * V[f,k]  -> K-vector
    T_lin = np.zeros((d, 256, C), np.float32)
    T_fac = np.zeros((d, 256, K), np.float32)
    off = 0
    for j in range(d):
        E = enc[j]
        dj = dims[j]
        T_lin[j] = E @ W[off:off+dj]
        T_fac[j] = E @ V[off:off+dj]
        off += dj

    # Pack weights: magic, d, K, C; T_lin[d][256][C], T_fac[d][256][K], b_lin[C], b_fac[K][C]
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_fac.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += b_fac.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index.
 * Weights format: magic 0x464D3130, d, K, C; T_lin[d][256][C], T_fac[d][256][K], b_lin[C], b_fac[K][C]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + C + (long)K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_fac = T_lin + (long)d * 256 * C;
    const float *b_lin = T_fac + (long)d * 256 * K;
    const float *b_fac = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *fac_acc = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) lin_acc[c] = b_lin[c];
        // Factor part: accumulate sum_v and sum_v2 per factor
        for (int kk = 0; kk < K; kk++) fac_acc[kk] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *tlin = T_lin + ((long)j * 256 + v) * C;
            const float *tfac = T_fac + ((long)j * 256 + v) * K;
            for (int c = 0; c < C; c++) lin_acc[c] += tlin[c];
            for (int kk = 0; kk < K; kk++) fac_acc[kk] += tfac[kk];
        }

        // Interaction: 0.5 * sum_k (fac_acc[k]^2) * b_fac[k]  -> but b_fac is K x C
        // Actually: interaction term per class c = 0.5 * sum_k (fac_acc[k]^2 - sum_j (tfac[j,k]^2)) * b_fac[k,c]
        // We didn't precompute sum of squares per column. But we can compute it on the fly:
        // Need sum over j of (tfac[j,k])^2. Let's compute it.
        // Alternative: store T_fac2[d][256][K] = tfac^2? Too much memory.
        // Instead, recompute sum of squares during inference:
        float fac_sq[K];
        for (int kk = 0; kk < K; kk++) fac_sq[kk] = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *tfac = T_fac + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float f = tfac[kk];
                fac_sq[kk] += f * f;
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                inter += (fac_acc[kk] * fac_acc[kk] - fac_sq[kk]) * b_fac[(long)kk * C + c];
            }
            scores[c] = lin_acc[c] + 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```