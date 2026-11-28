```hypothesis
Family: label-bigram (linear unigram + hashed adjacent bigrams). Character prediction depends heavily on local transitions (e.g. "th", "he"); a linear model on 16 independent one-hot positions misses these. Adding 15 hashed bigram features (4096 buckets) captures pairwise context with minimal parameters. Expect dev accuracy >0.37 (beats label-linear 0.3375) at ~12k Ir/item (2× label-linear cost). Failure: accuracy ≤0.34 or cost >20k Ir/item.
```
```python
import json, math, struct, sys

import numpy as np

B = 4096
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    T_uni = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    T_bi = rng.normal(0, 0.01, (B, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [T_uni, T_bi, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    mask = B - 1
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            bsize = len(yy)

            acc = np.tile(bias, (bsize, 1))
            for j in range(d):
                acc += T_uni[j, x[:, j], :]
            for j in range(d - 1):
                h = ((x[:, j].astype(np.uint32) * 257) + x[:, j + 1]) & mask
                acc += T_bi[h, :]

            acc_max = acc.max(axis=1, keepdims=True)
            exp_acc = np.exp(acc - acc_max)
            probs = exp_acc / exp_acc.sum(axis=1, keepdims=True)

            grad_acc = probs
            grad_acc[np.arange(bsize), yy] -= 1.0
            grad_acc /= bsize
            grad_acc = grad_acc.astype(np.float32)

            dbias = grad_acc.sum(axis=0)

            dT_uni = np.zeros_like(T_uni)
            for j in range(d):
                np.add.at(dT_uni[j], x[:, j], grad_acc)

            dT_bi = np.zeros_like(T_bi)
            for j in range(d - 1):
                h = ((x[:, j].astype(np.uint32) * 257) + x[:, j + 1]) & mask
                np.add.at(dT_bi, h, grad_acc)

            grads = [dT_uni, dT_bi, dbias]
            grads[0] += L2 * T_uni
            grads[1] += L2 * T_bi

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x4247524D
    out = struct.pack("<5i", magic, d, 0, C, B)
    out += T_uni.astype("<f4").tobytes()
    out += T_bi.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20 || hd[0] != 0x4247524D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], B = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * C + (long)B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_uni = (const float *)(hd + 5);
    const float *T_bi = T_uni + (long)d * 256 * C;
    const float *bias = T_bi + (long)B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    int mask = B - 1;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias[c];

        for (int j = 0; j < d; j++) {
            const float *t = T_uni + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        for (int j = 0; j < d - 1; j++) {
            uint32_t h = ((xi[j] * 257u) + xi[j + 1]) & mask;
            const float *t = T_bi + (long)h * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```