```hypothesis
Factorization Machine (FM) with 8-dimensional embeddings captures pairwise feature interactions crucial for tabular data like Adult, beating linear models at similar inference cost (O(K·d) with K=8). The dataset's mix of ordinal and categorical features benefits from learned interactions (e.g., education×occupation). If dev accuracy ≤0.8524 (linear) or inference cost >2500 Ir/item, the approach fails.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
SAMPLES = 1000000

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
    w0 = np.float32(0.0)
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            b = len(idx)

            w_vals = W[np.arange(d), xb.T]
            linear = w0 + w_vals.sum(axis=0)

            v_vals = V[np.arange(d), xb.T]
            sum_v = v_vals.sum(axis=0)
            sum_v2 = (v_vals * v_vals).sum(axis=0)
            interaction = 0.5 * ((sum_v * sum_v).sum(axis=1) - sum_v2.sum(axis=1))

            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)

            gw0 = grad.mean()

            gW = np.zeros_like(W)
            for j in range(d):
                np.add.at(gW[j], xb[:, j], grad)
            gW /= b
            gW += L2 * W

            gV = np.zeros_like(V)
            for j in range(d):
                other_sum = sum_v - v_vals[j]
                for i in range(b):
                    gV[j, xb[i, j]] += grad[i] * other_sum[i]
            gV /= b
            gV += L2 * V

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1**t)) / (np.sqrt(v_w0 / (1 - beta2**t)) + eps)

            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW * gW
            W -= LR * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += W.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int j = 0; j < K; j++) { sum_v[j] = 0.0f; sum_v2[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += W[(long)j * 256 + v];
            const float *vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = vj[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        }
        interaction *= 0.5f;

        float logit = linear + interaction;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char best = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(sum_v);
    free(sum_v2);
    return 0;
}
```