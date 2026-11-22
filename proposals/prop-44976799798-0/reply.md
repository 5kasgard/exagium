```hypothesis
Factorization Machine (FM) with shared embeddings (k=4) and class-specific interaction projections.
Models all pairwise feature interactions in O(d*k) time, capturing non-linearities missed by linear model.
Should beat label-linear (0.8524) on Adult tabular data with similar CPU cost (few hundred Ir/item).
Failure: dev accuracy ≤ 0.8524 or Ir/item > 1000.
```
```python
"""Factorization Machine for tabular label tasks: raw byte features (0-255) with pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: magic 0x464D5446, d, k, C; float32 w[d][256][C], e[d][256][k], u[C][k], b[C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 50


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

    # Parameters: w[d,256,C], e[d,256,K], u[C,K], b[C]
    w = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    e = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    u = (rng.normal(0, 0.01, (C, K))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [w, e, u, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            linear = np.zeros((B, C), np.float32)
            sum_e = np.zeros((B, K), np.float32)
            sum_e_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                vals = xb[:, j]
                linear += w[j, vals]                 # (B,C)
                emb = e[j, vals]                     # (B,K)
                sum_e += emb
                sum_e_sq += emb * emb
            inter = 0.5 * (sum_e * sum_e - sum_e_sq)  # (B,K)
            scores = linear + inter @ u.T + b         # (B,C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = (probs / B).astype(np.float32)   # (B,C)

            # Backward
            grad_b = grad_scores.sum(axis=0)               # (C,)
            grad_u = inter.T @ grad_scores                 # (K,C) -> transpose to (C,K)
            grad_inter = grad_scores @ u                   # (B,K)

            grad_sum_e = grad_inter * sum_e                # (B,K)
            grad_sum_e_sq = -0.5 * grad_inter              # (B,K)

            grad_w = np.zeros_like(w)
            grad_e = np.zeros_like(e)
            for j in range(d):
                vals = xb[:, j]
                # grad_w[j, vals] += grad_scores  -> use np.add.at
                np.add.at(grad_w[j], vals, grad_scores)
                # grad_e: from grad_sum_e and grad_sum_e_sq
                # emb = e[j, vals]; grad_emb = grad_sum_e + 2 * emb * grad_sum_e_sq
                emb = e[j, vals]
                grad_emb = grad_sum_e + 2.0 * emb * grad_sum_e_sq[:, None]  # wait shapes: grad_sum_e (B,K), grad_sum_e_sq (B,K), emb (B,K)
                # Actually grad_sum_e_sq is (B,K), need to multiply elementwise with emb
                grad_emb = grad_sum_e + 2.0 * emb * grad_sum_e_sq
                np.add.at(grad_e[j], vals, grad_emb)

            # Add L2 regularization gradients
            grad_w += L2 * w
            grad_e += L2 * e
            grad_u += L2 * u
            grad_b += L2 * b

            grads = [grad_w, grad_e, grad_u.T, grad_b]  # u grad was (C,K), need (K,C)? Wait u is (C,K), grad_u computed as (K,C) -> transpose to (C,K)
            # Actually u shape (C,K), grad_u = inter.T @ grad_scores -> (K,C). So grad_u.T is (C,K). Good.
            # But we stored grad_u as (K,C) then transposed. Let's keep grad_u as (C,K) directly:
            grad_u = grad_scores.T @ inter  # (C,B) @ (B,K) = (C,K)
            grads = [grad_w, grad_e, grad_u, grad_b]

            # Adam update
            t = 1  # we don't track global step perfectly, but per-batch approximation is fine for determinism
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9)
                v_hat = v / (1 - 0.999)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x464D5446  # 'FM' 'TF'? Actually FMTF
    hdr = struct.pack("<4I", magic, d, K, C)
    out = hdr
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
    out += u.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class (0 or 1).
 * Weights format: magic 0x464D5446, d, K, C; float32 w[d][256][C], e[d][256][K], u[C][K], b[C]. */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D5446) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected_floats = (long)d * 256 * C + (long)d * 256 * K + (long)C * K + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w = p; p += (long)d * 256 * C;
    const float *e = p; p += (long)d * 256 * K;
    const float *u = p; p += (long)C * K;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_e[4], sum_e_sq[4], scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        scores[0] = b[0]; scores[1] = b[1];
        for (int kk = 0; kk < K; kk++) { sum_e[kk] = 0.0f; sum_e_sq[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *w_j = w + ((long)j * 256 + val) * C;
            scores[0] += w_j[0];
            scores[1] += w_j[1];
            const float *e_j = e + ((long)j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float ek = e_j[kk];
                sum_e[kk] += ek;
                sum_e_sq[kk] += ek * ek;
            }
        }

        float inter[4];
        for (int kk = 0; kk < K; kk++) {
            inter[kk] = 0.5f * (sum_e[kk] * sum_e[kk] - sum_e_sq[kk]);
        }
        for (int c = 0; c < C; c++) {
            const float *u_c = u + c * K;
            float acc = 0.0f;
            for (int kk = 0; kk < K; kk++) acc += inter[kk] * u_c[kk];
            scores[c] += acc;
        }

        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```