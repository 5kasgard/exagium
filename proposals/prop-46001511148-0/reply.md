```hypothesis
Family: label-fm (Factorization Machine with 8-dim embeddings). 
Why: Adult has known feature interactions (e.g., education × occupation, capital-gain × hours); FM captures pairwise interactions with O(d*k) params and inference cost, beating linear while staying tiny. 
Failure signal: dev accuracy ≤ 0.8524 (linear) or instructions/item > 2× linear.
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
    assert C == 2
    rng = np.random.default_rng(SEED)

    # params: w0 (1), w[d,256], V[d,256,K]
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mw0, vw0 = np.float32(0.0), np.float32(0.0)
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mV, vV = np.zeros_like(V), np.zeros_like(V)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            # linear term
            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (bsz,)
            # interaction term: 0.5 * sum_f ( (sum_j V_j)^2 - sum_j V_j^2 )
            Vsel = V[np.arange(d), xb.T]  # (d, bsz, K)
            sumV = Vsel.sum(axis=0)       # (bsz, K)
            sumV2 = (Vsel * Vsel).sum(axis=0)
            inter = 0.5 * np.sum(sumV * sumV - sumV2, axis=1)  # (bsz,)
            score = lin + inter
            p = 1.0 / (1.0 + np.exp(-score))
            # gradient of loss w.r.t score
            g = (p - yb).astype(np.float32) / bsz  # (bsz,)

            # gradients
            # w0
            gw0 = np.sum(g) + L2 * w0
            # w
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], xb[:, j], g)
            gw += L2 * w
            # V
            gV = np.zeros_like(V)
            # d/dV_{j,v,f} = g_i * (sumV_{i,f} - V_{j,v,f})  for each sample i where xb[i,j]==v
            for j in range(d):
                vj = Vsel[j]  # (bsz, K)
                # for each feature j, we need to add g_i * (sumV_i - vj_i) to gV[j, xb[i,j]]
                # we can do: for each sample i, but vectorized:
                diff = (sumV - vj) * g[:, None]  # (bsz, K)
                # accumulate into gV[j, xb[:,j]]
                for f in range(K):
                    np.add.at(gV[j, :, f], xb[:, j], diff[:, f])
            gV += L2 * V

            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # w0
            mw0 = beta1 * mw0 + (1 - beta1) * gw0
            vw0 = beta2 * vw0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (mw0 / (1 - beta1 ** t)) / (np.sqrt(vw0 / (1 - beta2 ** t)) + eps)
            # w
            mw = beta1 * mw + (1 - beta1) * gw
            vw = beta2 * vw + (1 - beta2) * gw * gw
            w -= LR * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)
            # V
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV * gV
            V -= LR * (mV / (1 - beta1 ** t)) / (np.sqrt(vV / (1 - beta2 ** t)) + eps)

    # export: magic, d, K, C, w0, w, V
    magic = 0x464D4C31  # FML1
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sumV = malloc(sizeof(float) * K);
    float *sumV2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = w0;
        for (int f = 0; f < K; f++) { sumV[f] = 0.0f; sumV2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vjv[f];
                sumV[f] += val;
                sumV2[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sumV[f] * sumV[f] - sumV2[f];
        inter *= 0.5f;

        float score = lin + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```