```hypothesis
Family: label-fm (Factorization Machine). The Adult task has strong feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A rank-4 FM adds only ~14K latent parameters and captures all pairwise interactions via the efficient (∑v)²−∑v² trick. Expect ~0.860 dev accuracy at ~2000 Ir/item (3× linear cost, 4× lower than gbdt). Failure: accuracy ≤0.853 or Ir/item ≥3000.
```
```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 400000, 50

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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: bias, w[d,256], V[d,256,K]
    bias = np.float32(0.0)
    w = np.zeros((d, 256), np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mb, mw, mV = np.float32(0.0), np.zeros_like(w), np.zeros_like(V)
    vb, vw, vV = np.float32(0.0), np.zeros_like(w), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            bsz = len(yi)
            tgt = yi.astype(np.float32)

            # forward
            linear = np.full(bsz, bias, np.float32)
            sumV = np.zeros((bsz, K), np.float32)
            sumV2 = np.zeros((bsz, K), np.float32)
            for j in range(d):
                v = xi[:, j]
                linear += w[j, v]
                Vj = V[j, v]            # (bsz, K)
                sumV += Vj
                sumV2 += Vj * Vj
            inter = 0.5 * np.sum(sumV * sumV - sumV2, axis=1)
            score = linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad = (p - tgt).astype(np.float32)    # dL/dscore

            # backward
            dbias = float(grad.sum())
            dw = np.zeros_like(w)
            dV = np.zeros_like(V)
            for j in range(d):
                v = xi[:, j]
                np.add.at(dw[j], v, grad)
                # d(inter)/dV[j,v,k] = grad * (sumV[:,k] - V[j,v,k])
                for k in range(K):
                    dV[j, v, k] += grad * (sumV[:, k] - V[j, v, k])

            # Adam update
            t += 1
            # bias
            mb = beta1 * mb + (1 - beta1) * dbias
            vb = beta2 * vb + (1 - beta2) * (dbias * dbias)
            bias -= LR * (mb / (1 - beta1**t)) / (math.sqrt(vb / (1 - beta2**t)) + eps)

            # w
            mw = beta1 * mw + (1 - beta1) * dw
            vw = beta2 * vw + (1 - beta2) * (dw * dw)
            w -= LR * (mw / (1 - beta1**t)) / (np.sqrt(vw / (1 - beta2**t)) + eps)

            # V
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * (dV * dV)
            V -= LR * (mV / (1 - beta1**t)) / (np.sqrt(vV / (1 - beta2**t)) + eps)

            # L2 on V only (w is sparse, L2 not needed)
            if L2:
                V *= (1 - LR * L2)

    # export: magic, d, K, C; bias; w[d*256]; V[d*256*K]
    magic = 0x31204D46  # 'FM 1'
    C = 2
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr + struct.pack("<f", bias) + w.astype("<f4").tobytes() + V.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x31204D46) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K < 1 || K > 16 || C != 2) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias;
        float sumV[16], sumV2[16];
        for (int kk = 0; kk < K; kk++) { sumV[kk] = 0.0f; sumV2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                sumV[kk] += val;
                sumV2[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sumV[kk] * sumV[kk] - sumV2[kk];
        inter *= 0.5f;
        float score = linear + inter;

        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```