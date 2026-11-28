```hypothesis
We propose label-fm, a Factorization Machine with rank-4 latent vectors. It captures pairwise feature interactions crucial for tabular data like Adult, while remaining lightweight (~18k parameters). It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at similar or lower inference cost (O(d*r) per item). If dev accuracy does not exceed 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only. Exports: magic 0x464D3132, d, r, C, w0, linear[d*256], latent[d*256*r] (float32 little-endian)."""
import json, math, struct, sys

import numpy as np

RANK = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 50

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    w0 = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    latent = rng.normal(0, 0.01, (d, 256, RANK)).astype(np.float32)
    params = [w0, linear, latent]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(idx)

            # Forward
            # linear term: sum_j linear[j, xb[:,j]]
            lin_term = np.sum(linear[np.arange(d), xb.T], axis=0)  # (bsz,)
            # latent vectors for each feature: shape (d, bsz, RANK)
            vecs = latent[np.arange(d), xb.T]  # (d, bsz, RANK)
            sum_v = np.sum(vecs, axis=0)       # (bsz, RANK)
            sum_sq = np.sum(vecs * vecs, axis=0)
            fm_term = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)  # (bsz,)
            logit = w0 + lin_term + fm_term
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Gradient of loss w.r.t logit
            grad_logit = (prob - yb).astype(np.float32)  # (bsz,)

            # Gradients
            grad_w0 = np.mean(grad_logit)
            # linear gradients: for each feature j, scatter-add grad_logit to linear[j, xb[:,j]]
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], grad_logit)
            grad_linear /= bsz
            # latent gradients: for each feature j, grad_v = grad_logit[:,None] * (sum_v - vecs[j])
            grad_latent = np.zeros_like(latent)
            for j in range(d):
                diff = sum_v - vecs[j]  # (bsz, RANK)
                # grad_logit[:,None] * diff -> (bsz, RANK)
                g = grad_logit[:, None] * diff
                # scatter add to grad_latent[j, xb[:,j], :]
                for k in range(RANK):
                    np.add.at(grad_latent[j, :, k], xb[:, j], g[:, k])
            grad_latent /= bsz

            # L2 regularization
            grad_linear += L2 * linear
            grad_latent += L2 * latent

            grads = [grad_w0, grad_linear, grad_latent]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4if", 0x464D3132, d, RANK, C, w0)
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0/1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], r = hd[2], C = hd[3];
    if (C != 2 || d != 14 || r <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float *pf = (float *)(hd + 4);
    float w0 = pf[0];
    float *linear = pf + 1;
    float *latent = linear + (long)d * 256;
    long expected = 20 + 4L * ((long)d * 256 + (long)d * 256 * r);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * r);
    float *sum_sq = malloc(sizeof(float) * r);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin_sum = 0.0f;
        for (int kk = 0; kk < r; kk++) { sum_v[kk] = 0.0f; sum_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += linear[(long)j * 256 + v];
            float *vec = latent + ((long)j * 256 + v) * r;
            for (int kk = 0; kk < r; kk++) {
                float val = vec[kk];
                sum_v[kk] += val;
                sum_sq[kk] += val * val;
            }
        }
        float fm = 0.0f;
        for (int kk = 0; kk < r; kk++) {
            float sv = sum_v[kk];
            fm += sv * sv - sum_sq[kk];
        }
        fm *= 0.5f;
        float logit = w0 + lin_sum + fm;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```