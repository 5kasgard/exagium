```hypothesis
Family: label-fm (degree-2 Factorization Machine). Why: Optdigits pixels have strong local pairwise correlations (strokes, loops); FM captures these with O(k·d·C) params (k=2 factors) instead of O(d²). Inference ~3k mults/item (vs linear's table lookups); should lift accuracy 0.961→0.97+ while staying well below MLP cost. Failure: dev accuracy ≤ 0.9613 (interactions don't help this task).
```
```python
"""Factorization Machine (degree 2) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic, d, k, C; float32 mu[d], inv_sd[d], b[C], W[C][d], V[C][d][k]."""
import json, math, struct, sys

import numpy as np

K = 2
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # z-score each feature
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd
    Z = ((X.astype(np.float32) - mu) * inv_sd).astype(np.float32)  # (n, d)

    # Parameters per class: b[C], W[C,d], V[C,d,K]
    b = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (C, d)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / (d * K))).astype(np.float32)
    params = [b, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Z.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (B, d)
            yy = y[idx]         # (B,)

            # Forward: scores = b + z @ W.T + FM(z, V)
            # FM: for each class c, factor f: 0.5 * ( (sum_i V[c,i,f] z_i)^2 - sum_i (V[c,i,f] z_i)^2 )
            linear = z @ W.T + b  # (B, C)
            inter = np.zeros((len(yy), C), np.float32)
            for f in range(K):
                vf = V[:, :, f]           # (C, d)
                s = z @ vf.T              # (B, C)  sum_i V z_i
                sq = (z ** 2) @ (vf ** 2).T  # (B, C)  sum_i (V z_i)^2
                inter += 0.5 * (s * s - sq)
            scores = linear + inter

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = (probs / len(yy)).astype(np.float32)  # (B, C)

            # Gradients
            gb = grad_scores.sum(axis=0)                          # (C,)
            gW = grad_scores.T @ z + L2 * W                       # (C, d)
            gV = np.zeros_like(V)
            for f in range(K):
                vf = V[:, :, f]           # (C, d)
                s = z @ vf.T              # (B, C)
                # d(inter)/dV = z_i * (s_c - V_{c,i,f} z_i)
                # For each sample b, class c, feature i: z[b,i] * (s[b,c] - vf[c,i] * z[b,i])
                # Vectorized: gV[:,:,f] = z.T @ (grad_scores * (s - z @ vf.T?))
                # Actually: sum_b grad[b,c] * z[b,i] * (s[b,c] - vf[c,i] * z[b,i])
                # = z.T @ (grad_scores * s) - vf * (z.T @ (grad_scores * z))
                term1 = z.T @ (grad_scores * s)           # (d, C) -> (C, d) after .T
                term2 = vf * (z.T @ (grad_scores * z))    # (C, d) * (d, C).T -> (C, d)
                gV[:, :, f] = (term1.T - term2) + L2 * vf

            grads = [gb, gW, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, K, C; mu[d], inv_sd[d], b[C], W[C*d], V[C*d*K]
    magic = 0x31324D46  # 'FM21' little-endian
    out = struct.pack("<4i", magic, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += inv_sd.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte class index (0..C-1). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x31324D46) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (d != 64 || K < 1 || K > 8 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 2 * 4          /* mu, inv_sd */
        + (size_t)C * 4              /* b */
        + (size_t)C * d * 4          /* W */
        + (size_t)C * d * K * 4;     /* V */
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *mu = (const float *)(wb + 16);
    const float *inv_sd = mu + d;
    const float *b = inv_sd + d;
    const float *W = b + C;
    const float *V = W + (size_t)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);
    float *s = malloc(sizeof(float) * C);
    float *sq = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* z-score */
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) * inv_sd[j];

        /* linear part */
        for (int c = 0; c < C; c++) {
            float acc = b[c];
            const float *w = W + (size_t)c * d;
            for (int j = 0; j < d; j++) acc += w[j] * z[j];
            scores[c] = acc;
            s[c] = 0.0f;
            sq[c] = 0.0f;
        }

        /* FM interactions */
        for (int f = 0; f < K; f++) {
            const float *vf = V + (size_t)f * C * d;
            for (int c = 0; c < C; c++) {
                const float *v = vf + (size_t)c * d;
                float sum = 0.0f, sum_sq = 0.0f;
                for (int j = 0; j < d; j++) {
                    float vzi = v[j] * z[j];
                    sum += vzi;
                    sum_sq += vzi * vzi;
                }
                s[c] = sum;
                sq[c] = sum_sq;
            }
            for (int c = 0; c < C; c++) {
                scores[c] += 0.5f * (s[c] * s[c] - sq[c]);
            }
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```