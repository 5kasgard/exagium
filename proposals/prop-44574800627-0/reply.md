```hypothesis
Factorization Machine (FM) with per-byte-value embeddings (K=8) captures pairwise feature interactions crucial for Adult (e.g., education×occupation, marital-status×relationship). Lookup-table inference computes linear + FM interaction in O(d·K + d²·K) ≈ 260 ops/item, near label-linear speed. Fixed hyperparams (LR=3e-3, L2=1e-4, batch=256) chosen from tuning-split experiments. Fails if dev accuracy ≤ 0.8524 (label-linear) or CPU cost > 2× label-linear.
```
```python
"""Factorization Machine for tabular label tasks: per-byte linear + pairwise embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights: magic 0x464D3130, d, K, C; float32 T_lin[d][256][C], E[d][256][K], b[C]."""
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters: T_lin[d, 256, C], E[d, 256, K], b[C]
    scale = 0.01
    T_lin = (rng.normal(0, scale, (d, 256, C))).astype(np.float32)
    E = (rng.normal(0, scale, (d, 256, K))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [T_lin, E, b]
    state = [(np.zeros_like(p, dtype=np.float32), np.zeros_like(p, dtype=np.float32)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = X[idx]          # (B, d) uint8
            yy = y[idx]         # (B,) int64
            B = len(idx)

            # Forward
            linear = b + np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                v = z[:, j]
                linear += T_lin[j, v, :]
                e = E[j, v, :]
                sum_emb += e
                sum_emb_sq += e * e
            interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            logit = (linear[:, 1] - linear[:, 0]) + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            dlogit = p - yy

            # Gradients
            grads = [np.zeros_like(T_lin), np.zeros_like(E), np.zeros_like(b)]
            grads[2][1] += np.sum(dlogit)
            grads[2][0] -= np.sum(dlogit)
            for j in range(d):
                v = z[:, j]
                np.add.at(grads[0][j, :, 1], v, dlogit)
                np.add.at(grads[0][j, :, 0], v, -dlogit)
                diff = sum_emb - E[j, v, :]
                grad = dlogit[:, None] * diff
                np.add.at(grads[1][j], v, grad)

            # L2
            for i, p in enumerate(params):
                grads[i] += 2.0 * L2 * p

            # Adam
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1.0 - 0.9 ** t)
                v_hat = v / (1.0 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights: magic 0x464D3130, d, K, C; float32 T_lin[d][256][C], E[d][256][K], b[C]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64 ||
        nw != 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T_lin = (const float *)(hd + 4);
    const float *E = T_lin + (long)d * 256 * C;
    const float *b = E + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float acc0 = b[0], acc1 = b[1];
        for (int kk = 0; kk < K; kk++) { sum_emb[kk] = 0; sum_emb_sq[kk] = 0; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            acc0 += T_lin[((long)j * 256 + v) * C + 0];
            acc1 += T_lin[((long)j * 256 + v) * C + 1];
            const float *e = E + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }
        float interaction = 0;
        for (int kk = 0; kk < K; kk++)
            interaction += sum_emb[kk] * sum_emb[kk] - sum_emb_sq[kk];
        interaction *= 0.5f;
        acc1 += interaction;
        unsigned char best = (acc1 > acc0) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```