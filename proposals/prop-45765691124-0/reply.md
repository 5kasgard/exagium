```hypothesis
Factorization Machine (FM) with binary logit: treats each of the 14 bytes as a categorical field with 256 values, learns a linear weight per field-value plus a low-rank interaction matrix (rank K=8) shared across fields. Captures pairwise feature interactions missing in label-linear while keeping inference at O(d·K) per item. If dev accuracy does not exceed ~0.855 or inference cost exceeds label-mlp16, the family does not justify its complexity.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
LR, L2, BATCH, SAMPLES, MAX_EPOCHS, SEED = 0.01, 1e-4, 64, 600000, 100, 1


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
    n, d = X.shape
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)
            S = np.zeros((B, K), np.float32)
            Q = np.zeros((B, K), np.float32)
            for j in range(d):
                S += V[j, xb[:, j], :]
                Q += V[j, xb[:, j], :] ** 2
            inter = 0.5 * np.sum(S * S - Q, axis=1)
            logit = linear + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            g = p - yb

            gw0 = g.sum()
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], xb[:, j], g)
            gV = np.zeros_like(V)
            for j in range(d):
                for f in range(K):
                    term = g * (S[:, f] - V[j, xb[:, j], f])
                    np.add.at(gV[j, :, f], xb[:, j], term)

            gw0 += L2 * w0
            gw += L2 * w
            gV += L2 * V

            t += 1
            m_w0 = 0.9 * m_w0 + 0.1 * gw0
            v_w0 = 0.999 * v_w0 + 0.001 * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - 0.9 ** t)) / (math.sqrt(v_w0 / (1 - 0.999 ** t)) + 1e-8)

            m_w = 0.9 * m_w + 0.1 * gw
            v_w = 0.999 * v_w + 0.001 * gw * gw
            w -= LR * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)

            m_V = 0.9 * m_V + 0.1 * gV
            v_V = 0.999 * v_V + 0.001 * gV * gV
            V -= LR * (m_V / (1 - 0.9 ** t)) / (np.sqrt(v_V / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x464D3132
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1L + d * 256L + d * 256L * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + d * 256L;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float S[8], Q[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int f = 0; f < K; f++) { S[f] = 0.0f; Q[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vj[f];
                S[f] += val;
                Q[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += S[f] * S[f] - Q[f];
        inter *= 0.5f;
        float logit = linear + inter;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```