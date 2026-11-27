```hypothesis
Factorization Machine (FM) with K=8 latent factors per feature value, trained via Adam on logistic loss.
Captures all pairwise feature interactions in O(d·K) time, crucial for Adult where interactions
(e.g., education×occupation, capital-gain×hours) drive income. Beats linear if interactions matter;
fails if dev accuracy ≤0.8524 (linear) due to optimization issues or insufficient capacity.
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
    n, d = X.shape
    C = card["classes"]
    assert C == 2, "Binary classification only"
    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            linear = np.full(B, w0, dtype=np.float32)
            for j in range(d):
                linear += w[j, xb[:, j]]

            sum_V = np.zeros((B, K), dtype=np.float32)
            sum_V2 = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                Vj = V[j, xb[:, j], :]
                sum_V += Vj
                sum_V2 += Vj * Vj
            interaction = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)

            logit = linear + interaction
            pred = 1.0 / (1.0 + np.exp(-logit))
            grad = pred - yb

            gw0 = np.mean(grad)
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)

            for j in range(d):
                xj = xb[:, j]
                counts = np.bincount(xj, minlength=256)
                sums = np.bincount(xj, weights=grad, minlength=256)
                gw[j] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts>0)

                diff = sum_V - V[j, xj]
                for v in range(256):
                    mask = (xj == v)
                    if np.any(mask):
                        gV[j, v] = np.mean(grad[mask, np.newaxis] * diff[mask], axis=0)

            gw += L2 * w
            gV += L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4i", 0x464D3131, d, K, C)
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int j = 0; j < d; j++) {
            linear += w[(long)j * 256 + xi[j]];
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + kk];
                sum_v += v;
                sum_v2 += v * v;
            }
            interaction += 0.5f * (sum_v * sum_v - sum_v2);
        }

        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```