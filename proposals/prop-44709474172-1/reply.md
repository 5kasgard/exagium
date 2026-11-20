```hypothesis
Family: label-fm (Factorization Machine with shared interactions). The 8x8 digit pixels have strong spatial pairwise interactions (strokes, loops) that linear models miss. A degree-2 FM captures all pairwise interactions via a rank-k=2 factorization, adding only 128 params (64*2) and ~200 FLOPs/item. It should beat label-linear (0.9613) on accuracy while being faster (lower Ir/item). Failure: if dev accuracy ≤ 0.9613 or Ir/item ≥ 5058.
```

```python
import json, math, struct, sys
import numpy as np

K = 2
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    X = X_bytes.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    rng = np.random.default_rng(SEED)
    w0 = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (d, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, K))).astype(np.float32)

    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(scores, y_idx):
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs / len(y_idx)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = x.shape[0]

            linear = w0 + x @ w
            inter = np.zeros(B, np.float32)
            for f in range(K):
                Vf = V[:, f]
                xv = x @ Vf
                xv2 = x @ (Vf * Vf)
                inter += 0.5 * (xv * xv - xv2)
            scores = linear + inter[:, None]

            g = softmax_grad(scores, yy)
            g_sum = g.sum(axis=1)

            gw0 = g.sum(axis=0) + L2 * w0
            gw = x.T @ g + L2 * w
            gV = np.zeros_like(V)
            for f in range(K):
                Vf = V[:, f]
                xv = x @ Vf
                term1 = x.T @ (g_sum * xv)
                term2 = Vf * ((x * x).T @ g_sum)
                gV[:, f] = term1 - term2 + L2 * Vf

            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    out = struct.pack("<4i", 0x464D3132, d, K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += w0.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (2*d + C + d*C + d*K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mean = ptr; ptr += d;
    float *std = ptr; ptr += d;
    float *w0 = ptr; ptr += C;
    float *w = ptr; ptr += d * C;
    float *V = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(d * sizeof(float));
    float *linear = malloc(C * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) / std[j];
        }

        for (int c = 0; c < C; c++) {
            linear[c] = w0[c];
        }
        for (int j = 0; j < d; j++) {
            float xj = xnorm[j];
            for (int c = 0; c < C; c++) {
                linear[c] += xj * w[j * C + c];
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[j * K + f];
                float xv = xnorm[j] * v;
                sum1 += xv;
                sum2 += xv * v;
            }
            inter += 0.5f * (sum1 * sum1 - sum2);
        }

        for (int c = 0; c < C; c++) {
            scores[c] = linear[c] + inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```