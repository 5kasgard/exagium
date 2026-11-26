```hypothesis
Family: label-fm (Factorization Machine with per-position embeddings).
Why: Character prediction relies heavily on pairwise interactions (e.g., "th", "he", "qu"). FM captures all pairwise position interactions with O(d*K) parameters (d=16, K=16), adding only ~65K params over linear. The quadratic interaction term is computed in O(d*K + K*C) time, faster than MLP's ReLU layer. Beats linear on accuracy; matches or beats MLP16 on speed.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or latency > label-linear.
```

```python
import json, math, struct, sys

import numpy as np

K = 16
SEED, LR, L2, L2_EMB, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 1e-4, 64, 600000, 100


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

    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    W_emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    W_out = rng.normal(0, 0.01, (K, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [W_lin, W_emb, W_out, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(yy)

            lin_scores = b + np.sum(W_lin[np.arange(d)[:, None], x.T], axis=0)

            emb = W_emb[np.arange(d)[:, None], x.T].transpose(1, 0, 2)
            emb_sum = emb.sum(axis=1)
            emb_sq_sum = (emb**2).sum(axis=1)
            interaction = 0.5 * (emb_sum**2 - emb_sq_sum)
            fm_scores = interaction @ W_out

            scores = lin_scores + fm_scores
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_scores = probs / B

            grad_b = grad_scores.sum(axis=0)

            grad_W_lin = np.zeros_like(W_lin)
            for p in range(d):
                np.add.at(grad_W_lin[p], x[:, p], grad_scores)
            grad_W_lin += L2 * W_lin

            grad_W_out = interaction.T @ grad_scores
            grad_W_out += L2 * W_out

            grad_interaction = grad_scores @ W_out.T
            grad_W_emb = np.zeros_like(W_emb)
            for p in range(d):
                factor = (emb_sum - emb[:, p, :]) * grad_interaction
                np.add.at(grad_W_emb[p], x[:, p], factor)
            grad_W_emb += L2_EMB * W_emb

            grads = [grad_W_lin, grad_W_emb, grad_W_out, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += W_emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * K
        + 4L * K * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *W_emb = W_lin + (long)d * 256 * C;
    const float *W_out = W_emb + (long)d * 256 * K;
    const float *b = W_out + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = calloc(K, sizeof(float));
    float *emb_sq_sum = calloc(K, sizeof(float));
    float *scores = malloc(C * sizeof(float));
    float *interaction = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        memset(emb_sum, 0, K * sizeof(float));
        memset(emb_sq_sum, 0, K * sizeof(float));

        for (int p = 0; p < d; p++) {
            unsigned char byte = xi[p];
            const float *lin_row = W_lin + ((long)p * 256 + byte) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_row[c];

            const float *emb_row = W_emb + ((long)p * 256 + byte) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = emb_row[kk];
                emb_sum[kk] += v;
                emb_sq_sum[kk] += v * v;
            }
        }

        for (int kk = 0; kk < K; kk++) {
            interaction[kk] = 0.5f * (emb_sum[kk] * emb_sum[kk] - emb_sq_sum[kk]);
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum += interaction[kk] * W_out[(long)kk * C + c];
            }
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```