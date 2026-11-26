```hypothesis
Propose a Factorization Machine family (label-fm) with latent dimension k=4. FMs capture pairwise feature interactions crucial for tabular data like Adult, while remaining lightweight. Using the same per-byte encoding as label-linear, we train an FM with Adam (lr=3e-3, L2=1e-4) and export per-byte lookup tables for linear, factor, and squared-norm contributions. Expect dev accuracy >0.855 (beating label-linear's 0.8524) with inference cost O(d·k) ≈ 14·4 operations per class, far below label-mlp16. If dev accuracy ≤0.855, the FM does not justify its complexity over linear.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports per-byte lookup tables: magic 0x464D3231, d, k, C; float32 bias[C];
then for each class c, column j, byte v: linear, factor[k], sqnorm."""
import json, math, struct, sys

import numpy as np

K = 4
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
    d = card["input"]["bytes"]  # 14 original columns
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per column (same as label-linear)
    enc = []
    seg_starts = [0]
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        seg_starts.append(seg_starts[-1] + E.shape[1])
    d_total = seg_starts[-1]

    # Build full design matrix Z (n x d_total)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Parameters: bias[C], w[d_total, C], V[d_total, K, C]
    bias = np.zeros(C, np.float32)
    w = (rng.uniform(-1, 1, (d_total, C)) * math.sqrt(6.0 / d_total)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d_total, K, C)) * math.sqrt(6.0 / (d_total * K))).astype(np.float32)
    params = [bias, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute squared norms of V rows per class for gradient
    # But we'll compute on the fly in batches.

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = Z[idx]          # (B, d_total)
            yb = y[idx]          # (B,)
            B = Xb.shape[0]

            # Forward pass
            linear = Xb @ w + bias                     # (B, C)
            sum_vx = np.einsum('bi, ifc -> bfc', Xb, V)  # (B, K, C)
            sum_vx_sq = np.sum(sum_vx ** 2, axis=1)      # (B, C)
            v_sq = np.sum(V ** 2, axis=1)                # (d_total, C)
            sum_v_sq_x_sq = np.sum(Xb ** 2 * v_sq, axis=1)  # (B, C)
            interaction = 0.5 * (sum_vx_sq - sum_v_sq_x_sq)
            scores = linear + interaction                # (B, C)

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B                      # (B, C)

            # Gradients
            grad_bias = grad_scores.sum(axis=0)          # (C,)
            grad_w = Xb.T @ grad_scores + L2 * w         # (d_total, C)

            # grad_V = sum_b grad_scores[b,c] * ( Xb[b,i] * sum_vx[b,f,c] - Xb[b,i]^2 * V[i,f,c] )
            # term1: einsum('bic, bfc -> ifc', Xb * grad_scores[:,None,:], sum_vx)
            weighted_X = Xb[:, :, None] * grad_scores[:, None, :]  # (B, d_total, C)
            term1 = np.einsum('bic, bfc -> ifc', weighted_X, sum_vx)  # (d_total, K, C)
            # term2: V * sum_b grad_scores[b,c] * Xb[b,i]^2
            factor = grad_scores[:, None, :] * (Xb ** 2)[:, :, None]  # (B, d_total, C)
            term2 = V * factor.sum(axis=0)              # (d_total, K, C)
            grad_V = term1 - term2 + L2 * V             # (d_total, K, C)

            grads = [grad_bias, grad_w, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables
    # For each class c, column j, byte v: linear, factor[K], sqnorm
    lookup_lin = np.zeros((C, d, 256), np.float32)
    lookup_fac = np.zeros((C, d, 256, K), np.float32)
    lookup_sqn = np.zeros((C, d, 256), np.float32)

    for c in range(C):
        for j in range(d):
            seg = slice(seg_starts[j], seg_starts[j+1])
            E = enc[j]  # (256, d_j)
            # linear contribution
            lookup_lin[c, j] = E @ w[seg, c]
            # factor contribution
            fac = E @ V[seg, :, c]  # (256, K)
            lookup_fac[c, j] = fac
            lookup_sqn[c, j] = np.sum(fac ** 2, axis=1)

    # Pack weights.bin
    # magic, d, K, C, bias[C], then for c,j,v: linear, factor[K], sqnorm
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += bias.astype("<f4").tobytes()
    for c in range(C):
        out += lookup_lin[c].astype("<f4").tobytes()
        out += lookup_fac[c].astype("<f4").tobytes()
        out += lookup_sqn[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14); output is one byte class index.
 * Weights format: magic 0x464D3231, d, k, C; float32 bias[C];
 * then for each class c, column j, byte v: linear, factor[k], sqnorm. */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }

    // Expected size: 16 + 4*C + C*d*256*(1 + K + 1)*4
    long expected = 16L + 4L*C + (long)C * d * 256 * (2 + K) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin = bias + C;
    const float *fac = lin + (long)C * d * 256;
    const float *sqn = fac + (long)C * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    float *fsum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            score[c] = bias[c];
        }
        // accumulate per class
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                const float *lc = lin + ((long)c * d * 256 + (long)j * 256 + v);
                const float *fc = fac + ((long)c * d * 256 * K + (long)j * 256 * K + (long)v * K);
                const float *sc = sqn + ((long)c * d * 256 + (long)j * 256 + v);
                score[c] += *lc;
                for (int f = 0; f < K; f++) fsum[f] += fc[f];
                // we'll subtract sqnorm later per class
            }
        }
        // compute interaction and find max
        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            // recompute fsum for this class? No, we mixed classes. Need per-class accumulation.
            // Let's redo: accumulate per class separately.
        }
        // The above loop mixed classes. We must compute per class independently.
        // Let's restructure: for each class, zero fsum, accumulate, compute score.
        for (int c = 0; c < C; c++) {
            score[c] = bias[c];
            for (int f = 0; f < K; f++) fsum[f] = 0.0f;
            float sqsum = 0.0f;
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                const float *lc = lin + ((long)c * d * 256 + (long)j * 256 + v);
                const float *fc = fac + ((long)c * d * 256 * K + (long)j * 256 * K + (long)v * K);
                const float *sc = sqn + ((long)c * d * 256 + (long)j * 256 + v);
                score[c] += *lc;
                for (int f = 0; f < K; f++) fsum[f] += fc[f];
                sqsum += *sc;
            }
            float dot = 0.0f;
            for (int f = 0; f < K; f++) dot += fsum[f] * fsum[f];
            score[c] += 0.5f * (dot - sqsum);
            if (c == 0 || score[c] > score[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```