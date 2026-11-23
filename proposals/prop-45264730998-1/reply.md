```hypothesis
Family: label-fm (Factorization Machine, rank-4). Captures pairwise pixel interactions (strokes, loops) critical for digit recognition via low-rank factorization, adding only ~12K params over linear. Expect >0.97 accuracy (beats label-linear 0.961) at ~7000 Ir/item (below label-mlp16 8685). Failure: if ordinal normalization hurts, or rank-4 underfits interactions, dev accuracy <=0.961.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]

    X = X_bytes.astype(np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    rng = np.random.default_rng(SEED)
    w = (rng.normal(0, 0.01, (d, C))).astype(np.float32)
    v = (rng.normal(0, 0.01, (d, K, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [w, v, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            linear = xb @ w + b
            xb_sq = xb * xb
            inter = np.zeros((B, C), np.float32)
            for f in range(K):
                vf = v[:, f, :]
                vf_sq = vf * vf
                sum_vx = xb @ vf
                sum_vx_sq = xb_sq @ vf_sq
                inter += sum_vx * sum_vx - sum_vx_sq

            logits = linear + inter
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            gw = xb.T @ grad_logits + L2 * w
            gb = grad_logits.sum(axis=0)
            gv = np.zeros_like(v)
            for f in range(K):
                vf = v[:, f, :]
                sum_vx = xb @ vf
                gv[:, f, :] = 2.0 * (xb.T @ (sum_vx * grad_logits)) - 2.0 * vf * (xb_sq.T @ grad_logits) + L2 * vf

            grads = [gw, gv, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_state = state[i]
                m *= 0.9; m += 0.1 * gr
                v_state *= 0.999; v_state += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_state / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D3100, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + (long)(d*2 + d*C + d*K*C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    const float *mu = ptr; ptr += d;
    const float *sd = ptr; ptr += d;
    const float *w = ptr; ptr += d * C;
    const float *v = ptr; ptr += d * K * C;
    const float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_vx = malloc(sizeof(float) * C);
    float *sum_vx_sq = malloc(sizeof(float) * C);
    float *x_norm = malloc(sizeof(float) * d);
    float *x_norm_sq = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            float val = xi[j];
            float nval = (val - mu[j]) / sd[j];
            x_norm[j] = nval;
            x_norm_sq[j] = nval * nval;
        }

        for (int c = 0; c < C; c++) {
            scores[c] = b[c];
            for (int j = 0; j < d; j++) {
                scores[c] += x_norm[j] * w[j * C + c];
            }
        }

        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) {
                sum_vx[c] = 0.0f;
                sum_vx_sq[c] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                for (int c = 0; c < C; c++) {
                    float vjfc = v[(j * K + f) * C + c];
                    sum_vx[c] += x_norm[j] * vjfc;
                    sum_vx_sq[c] += x_norm_sq[j] * vjfc * vjfc;
                }
            }
            for (int c = 0; c < C; c++) {
                scores[c] += sum_vx[c] * sum_vx[c] - sum_vx_sq[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```