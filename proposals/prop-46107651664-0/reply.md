```hypothesis
Family: label-fm (Factorization Machine with 8-dimensional latent vectors).
Why: Adult has 14 mixed-type features; a low-rank pairwise interaction model captures
     feature crosses that linear models miss, while staying tiny (≈130 KB) and fast
     (O(d·k) per item). FMs are strong on tabular data with categorical features.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or latency ≥ label-linear.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3101, d, k, C; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = xb.shape[0]

            # Forward
            linear = w0 + np.sum(w[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
            sum_v = np.zeros((bsz, K), dtype=np.float32)
            sum_v2 = np.zeros((bsz, K), dtype=np.float32)
            for j in range(d):
                vj = V[j, xb[:, j]]          # (bsz, K)
                sum_v += vj
                sum_v2 += vj * vj
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            score = linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb).astype(np.float32)  # (bsz,)

            # Gradients
            gw0 = np.mean(g)
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)

            for j in range(d):
                xj = xb[:, j]
                # w gradient
                for v in range(256):
                    mask = (xj == v)
                    if np.any(mask):
                        gw[j, v] = np.mean(g[mask])
                # V gradient
                vj = V[j, xj]                 # (bsz, K)
                # sum_v[f] - vj[:,f] for each sample
                diff = sum_v - vj              # (bsz, K)
                for f in range(K):
                    for v in range(256):
                        mask = (xj == v)
                        if np.any(mask):
                            gV[j, v, f] = np.mean(g[mask] * diff[mask, f])

            # L2
            gw0 += L2 * w0
            gw += L2 * w
            gV += L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            lr_t = LR * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= lr_t * m_w0 / (math.sqrt(v_w0) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= lr_t * m_w / (np.sqrt(v_w) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= lr_t * m_V / (np.sqrt(v_V) + eps)

    # Write weights
    magic = 0x464D3101
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3101) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += (long)d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *vj = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = vj[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        float score = linear + inter;
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```