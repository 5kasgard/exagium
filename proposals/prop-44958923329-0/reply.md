```hypothesis
label-fm: Factorization Machine (rank 4) on the original 14 quantized features.
Captures all pairwise feature interactions with O(d*K) params and inference cost,
where linear models miss interactions and MLPs cost more. Adult has strong
feature interactions (education × occupation, capital-gain × hours, etc.).
Expected: >0.855 dev accuracy at <1000 Ir/item. If accuracy ≤ label-linear,
interactions don't help or rank 4 is too low.
```
```python
import json, math, struct, sys
import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, K = 1, 0.01, 1e-4, 64, 600000, 100, 4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.frombuffer(b"".join(records(sys.argv[3])), dtype=np.uint8).astype(np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            # Forward
            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)
            inter = np.zeros(B, dtype=np.float32)
            for f in range(K):
                Vf = V[:, :, f]
                vals = Vf[np.arange(d), xb.T]
                sum_V = np.sum(vals, axis=0)
                sum_V2 = np.sum(vals * vals, axis=0)
                inter += 0.5 * (sum_V * sum_V - sum_V2)
            score = linear + inter

            sig = 1.0 / (1.0 + np.exp(-score))
            grad = (sig - yb).astype(np.float32)

            # Gradients
            gw0 = np.mean(grad) + L2 * w0

            gw = np.zeros_like(w)
            for j in range(d):
                gw[j] = np.bincount(xb[:, j], weights=grad, minlength=256) / B + L2 * w[j]

            gV = np.zeros_like(V)
            for f in range(K):
                Vf = V[:, :, f]
                vals_f = Vf[np.arange(d), xb.T]
                sum_Vf = np.sum(vals_f, axis=0)
                for j in range(d):
                    vj = xb[:, j]
                    Vjvf = Vf[j, vj]
                    gV_jf = grad * (sum_Vf - Vjvf)
                    gV[j, :, f] = np.bincount(vj, weights=gV_jf, minlength=256) / B + L2 * V[j, :, f]

            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Save: magic, d, K, C, w0, w[d*256], V[d*256*K]
    out = struct.pack("<4i", 0x464D31, d, K, C)
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float w0 = *ptr++;
    float *w = ptr; ptr += d * 256;
    float *V = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) score += w[j * 256 + xi[j]];

        for (int f = 0; f < K; f++) {
            float sum_V = 0.0f, sum_V2 = 0.0f;
            float *Vf = V + f * d * 256;
            for (int j = 0; j < d; j++) {
                float v = Vf[j * 256 + xi[j]];
                sum_V += v;
                sum_V2 += v * v;
            }
            score += 0.5f * (sum_V * sum_V - sum_V2);
        }
        best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```