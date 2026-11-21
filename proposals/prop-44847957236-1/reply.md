```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Digits are defined by stroke intersections and curves; pairwise pixel products capture these interactions linearly. Beats label-linear (0.961) by adding 2080 quadratic features (2144 total), still far cheaper than GBDT/MLP at inference.
Failure: Dev accuracy ≤ 0.962 or Ir/item ≥ 50000 (cost exceeds GBDT without accuracy gain).
```

```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(Z):
    """Z: (n, d) -> (n, D) where D = d + d*(d+1)/2"""
    n, d = Z.shape
    D = d + d * (d + 1) // 2
    Phi = np.zeros((n, D), dtype=np.float32)
    Phi[:, :d] = Z
    idx = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    return Phi


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    D = d + d * (d + 1) // 2

    rng = np.random.default_rng(SEED)

    # Z-score per feature
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # Expand to polynomial features
    Phi = expand(Z)
    n, D_ = Phi.shape
    assert D_ == D

    # Initialize weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]

            S = phi @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)

            grad_W = phi.T @ G + L2 * W
            grad_b = G.sum(0)
            grads = [grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save weights: magic, d, D, C, mu[d], sd[d], W[D*C], b[C]
    out = struct.pack("<4i", 0x504F4C59, d, D, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 8L * d + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mu = (float *)(hd + 4);
    float *sd = mu + d;
    float *W = sd + d;
    float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) / sd[j];

        int idx = 0;
        for (int j = 0; j < d; j++) phi[idx++] = z[j];
        for (int j = 0; j < d; j++)
            for (int k2 = j; k2 < d; k2++)
                phi[idx++] = z[j] * z[k2];
        if (idx != D) { fprintf(stderr, "expansion size mismatch\n"); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < D; k++) {
            float phik = phi[k];
            if (phik == 0.0f) continue;
            for (int c = 0; c < C; c++)
                scores[c] += phik * W[(long)k * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```