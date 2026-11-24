```hypothesis
Family: label-fm (Factorization Machine with 8-dim embeddings). 
Why: Adult has strong pairwise interactions (education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions with only O(d*K) params (d=14 features, K=8), adding ~115K weights vs linear's ~3.5K, but inference stays fast (O(d*K) per item). 
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)
            V_x = V[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_V = np.sum(V_x, axis=1)
            sum_V_sq = np.sum(V_x ** 2, axis=1)
            interaction = 0.5 * np.sum(sum_V ** 2 - sum_V_sq, axis=1)
            logits = linear + interaction

            sigmoid = 1.0 / (1.0 + np.exp(-logits))
            dlogits = (sigmoid - yb) / B

            grad_w0 = np.sum(dlogits)
            grad_w = np.zeros_like(w)
            for j in range(d):
                np.add.at(grad_w[j], xb[:, j], dlogits)

            grad_V = np.zeros_like(V)
            for j in range(d):
                factor = dlogits[:, np.newaxis] * (sum_V - V_x[:, j, :])
                for k in range(K):
                    np.add.at(grad_V[j, :, k], xb[:, j], factor[:, k])

            grad_w0 += L2 * w0
            grad_w += L2 * w
            grad_V += L2 * V

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(hd + 4);
    float w0 = ptr[0]; ptr++;
    const float *w = ptr; ptr += (long)d * 256;
    const float *V = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int j = 0; j < d; j++) linear += w[j * 256 + xi[j]];

        float sum_V[64], sum_V_sq[64];
        for (int kk = 0; kk < K; kk++) { sum_V[kk] = 0.0f; sum_V_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = v[kk];
                sum_V[kk] += val;
                sum_V_sq[kk] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) interaction += sum_V[kk] * sum_V[kk] - sum_V_sq[kk];
        interaction *= 0.5f;

        float logit = linear + interaction;
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```