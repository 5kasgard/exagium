```hypothesis
Family: label-fm (Factorization Machine with K=8 embeddings). Adult income benefits from pairwise feature interactions (e.g., education × occupation) which a linear model misses. FM adds a cheap interaction term (O(d·K)) that captures these while staying faster than an MLP. Expected dev accuracy >0.855 (beating label-linear's 0.8524) with Ir/item well below label-mlp16's 2325. If accuracy ≤0.855 or Ir/item ≥2000, the hypothesis fails.
```
```python
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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: linear weights (d,256,C), embeddings (d,256,C,K), biases (C)
    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    E_emb = rng.normal(0, 0.01, (d, 256, C, K)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [W_lin, E_emb, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    t = 0

    for epoch in range(MAX_EPOCHS):
        if t >= total_steps:
            break
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            if t >= total_steps:
                break
            idx = order[s:s + BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            # Forward
            linear_sum = np.zeros((B, C), np.float32)
            for j in range(d):
                linear_sum += W_lin[j, x_batch[:, j], :]

            S = np.zeros((B, C, K), np.float32)
            sum_sq = np.zeros((B, C, K), np.float32)
            for c in range(C):
                for j in range(d):
                    emb = E_emb[j, x_batch[:, j], c, :]
                    S[:, c, :] += emb
                    sum_sq[:, c, :] += emb * emb
            interaction = 0.5 * np.sum(S * S - sum_sq, axis=2)

            scores = b + linear_sum + interaction

            # Softmax + gradient
            scores_max = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            grad_scores = probs.copy()
            grad_scores[np.arange(B), y_batch] -= 1.0
            grad_scores /= B

            # Gradients
            grad_b = np.sum(grad_scores, axis=0)

            grad_W_lin = np.zeros_like(W_lin)
            for j in range(d):
                for c in range(C):
                    np.add.at(grad_W_lin[j, :, c], x_batch[:, j], grad_scores[:, c])

            grad_E_emb = np.zeros_like(E_emb)
            for c in range(C):
                for j in range(d):
                    xj = x_batch[:, j]
                    for b_idx in range(B):
                        val = xj[b_idx]
                        g = grad_scores[b_idx, c] * (S[b_idx, c, :] - E_emb[j, val, c, :])
                        grad_E_emb[j, val, c, :] += g

            # L2 regularization
            grad_W_lin += L2 * W_lin
            grad_E_emb += L2 * E_emb

            # Adam update
            grads = [grad_W_lin, grad_E_emb, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic, d, K, C; then W_lin, E_emb, b (all float32 little-endian)
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out_bytes = header
    out_bytes += W_lin.astype("<f4").tobytes()
    out_bytes += E_emb.astype("<f4").tobytes()
    out_bytes += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```
```c
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
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * C + (long)d * 256 * C * K + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *E_emb = W_lin + (long)d * 256 * C;
    const float *b = E_emb + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float score[2];
    float sum_e[8], sum_sq[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        score[0] = b[0];
        score[1] = b[1];

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *w = W_lin + ((long)j * 256 + v) * C;
            score[0] += w[0];
            score[1] += w[1];
        }

        for (int c = 0; c < C; c++) {
            for (int k_idx = 0; k_idx < K; k_idx++) {
                sum_e[k_idx] = 0.0f;
                sum_sq[k_idx] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                const float *e = E_emb + (((long)j * 256 + v) * C + c) * K;
                for (int k_idx = 0; k_idx < K; k_idx++) {
                    float val = e[k_idx];
                    sum_e[k_idx] += val;
                    sum_sq[k_idx] += val * val;
                }
            }
            float inter = 0.0f;
            for (int k_idx = 0; k_idx < K; k_idx++) {
                inter += sum_e[k_idx] * sum_e[k_idx] - sum_sq[k_idx];
            }
            inter *= 0.5f;
            score[c] += inter;
        }

        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```