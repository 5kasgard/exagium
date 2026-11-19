```hypothesis
Family: label-fm (Factorization Machine, rank-4). Optdigits pixels exhibit local spatial correlations; FM captures pairwise interactions with O(d·k) parameters (k=4) instead of O(d²). Per-class factor matrices add only ~3K params. Expect >0.965 dev accuracy at ~10K Ir/item. Fails if accuracy ≤ 0.9613 (label-linear) or cost > 15K Ir/item.
```
```python
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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).astype(np.float32)

    bias = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (C, d)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / d)).astype(np.float32)

    params = [bias, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X_norm.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_norm[idx]
            yy = y[idx]
            B = len(yy)

            linear = x @ W.T + bias
            inter = np.zeros((B, C), np.float32)
            for c in range(C):
                Vc = V[c]
                xV = x @ Vc
                x2 = x * x
                Vc2 = Vc * Vc
                sum_x2_Vc2 = x2 @ Vc2
                inter[:, c] = 0.5 * np.sum(xV * xV - sum_x2_Vc2, axis=1)

            scores = linear + inter
            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            probs /= B

            grad_bias = probs.sum(axis=0)
            grad_W = probs.T @ x + L2 * W
            grad_V = np.zeros_like(V)
            for c in range(C):
                Vc = V[c]
                pc = probs[:, c:c+1]
                xV = x @ Vc
                pcx = pc * x
                x2 = x * x
                term1 = pcx.T @ xV
                term2 = Vc * (pc * x2).sum(axis=0)[:, None]
                grad_V[c] = term1 - term2 + L2 * Vc

            grads = [grad_bias, grad_W, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x464D3131
    out = struct.pack("<4i", magic, d, K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    for c in range(C):
        out += np.array([bias[c]], dtype="<f4").tobytes()
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()

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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || K != 4 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }

    long expected = 16 + 2L * d * 4 + (long)C * (4 + (long)d * 4 + (long)d * K * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mean = ptr; ptr += d;
    float *std = ptr; ptr += d;

    float *bias = malloc(C * sizeof(float));
    float *W = malloc((long)C * d * sizeof(float));
    float *V = malloc((long)C * d * K * sizeof(float));

    for (int c = 0; c < C; c++) {
        bias[c] = *ptr++;
        for (int j = 0; j < d; j++) W[(long)c * d + j] = *ptr++;
        for (int j = 0; j < d; j++) for (int f = 0; f < K; f++) V[((long)c * d + j) * K + f] = *ptr++;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_norm = malloc(d * sizeof(float));
    float *x2 = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            float val = (float)xi[j];
            x_norm[j] = (val - mean[j]) / std[j];
            x2[j] = x_norm[j] * x_norm[j];
        }

        for (int c = 0; c < C; c++) {
            float linear = bias[c];
            for (int j = 0; j < d; j++) linear += x_norm[j] * W[(long)c * d + j];

            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sum_vx = 0.0f, sum_v2x2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = V[((long)c * d + j) * K + f];
                    sum_vx += v * x_norm[j];
                    sum_v2x2 += v * v * x2[j];
                }
                inter += 0.5f * (sum_vx * sum_vx - sum_v2x2);
            }
            scores[c] = linear + inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_norm); free(x2); free(scores);
    free(bias); free(W); free(V);
    free(hd);
    if (x) free(x);
    return 0;
}
```