```hypothesis
Family: label-fm (Factorization Machine, degree 2). Uses 14 raw byte features (256 values each) with per-class linear weights and K=8 factor vectors. Captures pairwise interactions (e.g., occupation×education) crucial for Adult, while inference stays O(C·d·K)≈224 ops/item. Expected to beat label-linear (0.8524); if dev accuracy ≤0.853, interactions don't help here.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 256, 600000, 100


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

    w = np.zeros((C, d, 256), dtype=np.float32)
    f = (rng.normal(0, 0.01 / math.sqrt(K), (C, d, 256, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [w, f, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            scores = bias[:, None].astype(np.float32)
            for c in range(C):
                for j in range(d):
                    scores[c] += w[c, j, xb[:, j]]

            for c in range(C):
                sum1 = np.zeros((m, K), dtype=np.float32)
                sum2 = np.zeros((m, K), dtype=np.float32)
                for j in range(d):
                    fj = f[c, j, xb[:, j], :]
                    sum1 += fj
                    sum2 += fj * fj
                inter = 0.5 * np.sum(sum1 * sum1 - sum2, axis=1)
                scores[c] += inter

            scores = scores.T
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            grad_scores = probs / m

            grad_w = np.zeros_like(w)
            grad_f = np.zeros_like(f)
            grad_bias = np.zeros_like(bias)

            for c in range(C):
                gc = grad_scores[:, c]
                grad_bias[c] = gc.sum()
                for j in range(d):
                    np.add.at(grad_w[c, j], xb[:, j], gc)

                sum1 = np.zeros((m, K), dtype=np.float32)
                for j in range(d):
                    sum1 += f[c, j, xb[:, j], :]

                for j in range(d):
                    fj_vals = f[c, j, xb[:, j], :]
                    contrib = (sum1 - fj_vals) * gc[:, None]
                    for l in range(K):
                        np.add.at(grad_f[c, j, :, l], xb[:, j], contrib[:, l])

            grad_w += L2 * w
            grad_f += L2 * f

            t += 1
            grads = [grad_w, grad_f, grad_bias]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_adam, v_adam = state[i]
                m_adam *= 0.9; m_adam += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m_adam / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += f.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights params\n"); return 3;
    }
    long expected = 16L + 4L*C + 4L*C*d*256 + 4L*C*d*256*K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + C;
    const float *f = w + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum1 = malloc(sizeof(float) * K);
    float *sum2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = bias[c];
            for (int j = 0; j < d; j++) {
                scores[c] += w[((long)c * d + j) * 256 + xi[j]];
            }
            for (int l = 0; l < K; l++) {
                sum1[l] = 0.0f;
                sum2[l] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                const float *fj = f + (((long)c * d + j) * 256 + v) * K;
                for (int l = 0; l < K; l++) {
                    float fv = fj[l];
                    sum1[l] += fv;
                    sum2[l] += fv * fv;
                }
            }
            for (int l = 0; l < K; l++) {
                scores[c] += 0.5f * (sum1[l] * sum1[l] - sum2[l]);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```