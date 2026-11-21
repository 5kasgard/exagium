```hypothesis
Family: label-fm (Factorization Machine with shared latent vectors).
Why: Adult has 14 mixed ordinal/categorical features; a linear model misses pairwise interactions that FM captures with O(d*K) extra params (K=8). Inference adds only ~14*K mults/item, far cheaper than MLP. If dev accuracy ≤ label-linear (+0.5%) or latency ≥ 2× label-linear, the family does not justify itself.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, K, C; float32 T[d][256][C], V[d][256][K], W_inter[C][K], b[C]."""
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

    # parameters
    T = np.zeros((d, 256, C), np.float32)          # linear weights
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)  # latent vectors
    W_inter = np.zeros((C, K), np.float32)         # interaction projection
    b = np.zeros(C, np.float32)

    params = [T, V, W_inter, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            # linear: (bs, C)
            linear = b.copy()
            for j in range(d):
                linear += T[j, xb[:, j], :]

            # FM interaction vector: (bs, K)
            s_vec = np.zeros((bs, K), np.float32)
            sq_vec = np.zeros((bs, K), np.float32)
            for j in range(d):
                v = V[j, xb[:, j], :]
                s_vec += v
                sq_vec += v * v
            inter = 0.5 * (s_vec * s_vec - sq_vec)

            scores = linear + inter @ W_inter.T

            # softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # gradients
            g_T = np.zeros_like(T)
            g_V = np.zeros_like(V)
            g_W = np.zeros_like(W_inter)
            g_b = probs.sum(axis=0)

            for j in range(d):
                xj = xb[:, j]
                # linear part
                for c in range(C):
                    g_T[j, xj, c] += probs[:, c].sum()
                # interaction part
                # d(inter)/dV = (s_vec - V) * W_inter[c] * probs[:,c] summed over c
                # we can compute per-class contribution then accumulate
                for c in range(C):
                    w = W_inter[c]
                    coef = probs[:, c][:, None] * w  # (bs, K)
                    # d inter / d V = s_vec - V  (since inter = 0.5*(s^2 - sq), derivative w.r.t V is s - V)
                    grad_v = coef * (s_vec - V[j, xj, :])
                    g_V[j, xj, :] += grad_v
                    g_W[c] += (inter * probs[:, c][:, None]).sum(axis=0)

            grads = [g_T, g_V, g_W, g_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += T.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W_inter.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax).
 * Weights layout: magic, d, K, C; T[d][256][C], V[d][256][K], W_inter[C][K], b[C] (all float32 LE). */
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
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)
        + (size_t)d * 256 * K * sizeof(float)
        + (size_t)C * K * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *V = T + (size_t)d * 256 * C;
    const float *W_inter = V + (size_t)d * 256 * K;
    const float *b = W_inter + (size_t)C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_vec = malloc(sizeof(float) * K);
    float *sq_vec = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* linear part */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((size_t)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tj[c];
        }

        /* FM interaction */
        for (int u = 0; u < K; u++) { s_vec[u] = 0.0f; sq_vec[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((size_t)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float v = vj[u];
                s_vec[u] += v;
                sq_vec[u] += v * v;
            }
        }
        for (int u = 0; u < K; u++) inter[u] = 0.5f * (s_vec[u] * s_vec[u] - sq_vec[u]);

        for (int c = 0; c < C; c++) {
            const float *wc = W_inter + (size_t)c * K;
            float sum = 0.0f;
            for (int u = 0; u < K; u++) sum += inter[u] * wc[u];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```