```hypothesis
Family: label-fm (Factorization Machine). 
Why: Adult has mixed ordinal/categorical features where pairwise interactions (e.g., education × occupation) are predictive. A rank-8 FM adds only ~65k parameters over linear, learns per-byte embeddings, and computes interactions in O(d·r) at inference. 
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```

```python
import json, math, struct, sys
import numpy as np

R = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    bias = np.zeros(C, np.float32)
    lin = np.zeros((d, 256, C), np.float32)
    V = rng.normal(0, 0.01, (d, 256, C, R)).astype(np.float32)

    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(idx)

            lin_b = lin[np.arange(d), Xb.T].transpose(1,0,2)
            V_b = V[np.arange(d), Xb.T].transpose(1,0,2,3)

            S = V_b.sum(axis=1)
            V_sq_sum = (V_b * V_b).sum(axis=1)
            linear = lin_b.sum(axis=1)
            inter = 0.5 * ((S * S - V_sq_sum).sum(axis=2))
            logits = bias + linear + inter

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            g_bias = grad_logits.sum(axis=0)
            g_lin = np.zeros_like(lin)
            g_V = np.zeros_like(V)
            for j in range(d):
                v_idx = Xb[:, j]
                np.add.at(g_lin[j], v_idx, grad_logits)
                V_j = V_b[:, j]
                g_V_j = grad_logits[:,:,None] * (S - V_j)
                np.add.at(g_V[j], v_idx, g_V_j)

            g_bias += L2 * bias
            g_lin += L2 * lin
            g_V += L2 * V

            t += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            m_bias = b1*m_bias + (1-b1)*g_bias; v_bias = b2*v_bias + (1-b2)*g_bias*g_bias
            bias -= LR * (m_bias/(1-b1**t)) / (np.sqrt(v_bias/(1-b2**t)) + eps)
            m_lin = b1*m_lin + (1-b1)*g_lin; v_lin = b2*v_lin + (1-b2)*g_lin*g_lin
            lin -= LR * (m_lin/(1-b1**t)) / (np.sqrt(v_lin/(1-b2**t)) + eps)
            m_V = b1*m_V + (1-b1)*g_V; v_V = b2*v_V + (1-b2)*g_V*g_V
            V -= LR * (m_V/(1-b1**t)) / (np.sqrt(v_V/(1-b2**t)) + eps)

    out = struct.pack("<4i", 0x464D3130, d, R, C)
    out += bias.astype("<f4").tobytes()
    for j in range(d):
        for v in range(256):
            out += lin[j, v].astype("<f4").tobytes()
            out += V[j, v].astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1 || R > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L*C + 4L*d*256*C*(1+R);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin = bias + C;
    const float *V = lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float acc[256];
        for (int c = 0; c < C; c++) acc[c] = bias[c];

        float V_sum[256][256], V_sq_sum[256][256];
        for (int c = 0; c < C; c++)
            for (int f = 0; f < R; f++)
                V_sum[c][f] = V_sq_sum[c][f] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long lbase = (long)j * 256 * C + (long)v * C;
            for (int c = 0; c < C; c++) acc[c] += lin[lbase + c];
            long vbase = (long)j * 256 * C * R + (long)v * C * R;
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < R; f++) {
                    float val = V[vbase + c * R + f];
                    V_sum[c][f] += val;
                    V_sq_sum[c][f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++)
            for (int f = 0; f < R; f++)
                acc[c] += 0.5f * (V_sum[c][f] * V_sum[c][f] - V_sq_sum[c][f]);

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```