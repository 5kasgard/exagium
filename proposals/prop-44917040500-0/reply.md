```hypothesis
Family: label-fm (Factorization Machine with per-feature latent vectors).
Why: Adult income depends on feature interactions (education×occupation, marital-status×relationship) that linear models miss; FM models all pairwise interactions via factorized vectors, adding only O(d·k·C) params and O(d·k·C) inference cost. With k=16 it should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower cost than GBDT.
Falsified if: dev accuracy ≤ 0.8524 (label-linear baseline).
```
```python
import json, math, struct, sys
import numpy as np

K = 16
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
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    W_int = rng.normal(0, 0.01, (d, 256, K, C)).astype(np.float32)

    params = [W_lin, b_lin, W_int]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            scores = np.tile(b_lin, (B, 1))
            for j in range(d):
                scores += W_lin[j, xb[:, j], :]

            sum_v = np.zeros((B, K, C), np.float32)
            sum_v_sq = np.zeros((B, K, C), np.float32)
            for j in range(d):
                v_feat = W_int[j, xb[:, j], :, :]
                sum_v += v_feat
                sum_v_sq += v_feat * v_feat
            scores += 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)

            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads = probs / B

            g_b = grads.sum(axis=0)
            g_Wlin = np.zeros_like(W_lin)
            for j in range(d):
                for c in range(C):
                    np.add.at(g_Wlin[j, :, c], xb[:, j], grads[:, c])

            g_Wint = np.zeros_like(W_int)
            for j in range(d):
                v_feat = W_int[j, xb[:, j], :, :]
                diff = sum_v - v_feat
                weighted_diff = diff * grads[:, None, :]
                for c in range(C):
                    for f in range(K):
                        np.add.at(g_Wint[j, :, f, c], xb[:, j], weighted_diff[:, f, c])

            t += 1
            for i, (p, gr) in enumerate(zip(params, [g_Wlin, g_b, g_Wint])):
                gr += L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x464D31
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += W_int.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *b_lin = p; p += C;
    const float *W_int = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K * C);
    float *sum_v_sq = malloc(sizeof(float) * K * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b_lin[c];

        for (int j = 0; j < d; j++) {
            const float *w = W_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        for (int c = 0; c < C; c++)
            for (int f = 0; f < K; f++)
                sum_v[c * K + f] = sum_v_sq[c * K + f] = 0.0f;

        for (int j = 0; j < d; j++) {
            const float *v = W_int + ((long)j * 256 + xi[j]) * K * C;
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < K; f++) {
                    float val = v[f * C + c];
                    sum_v[c * K + f] += val;
                    sum_v_sq[c * K + f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) {
                float sv = sum_v[c * K + f];
                scores[c] += 0.5f * (sv * sv - sum_v_sq[c * K + f]);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```