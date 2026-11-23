```hypothesis
Family: label-rff (Random Fourier Features approximating RBF kernel + linear classifier).
Why: Optdigits pixels have spatial structure; RBF kernel captures locality but exact kernel methods are slow.
RFF with M=128 random features gives a linear model in expanded space that approximates the RBF kernel,
typically beating plain linear on vision tasks while staying faster than an MLP (only cos + matvec).
Failure signal: dev accuracy ≤ 0.962 (no meaningful gain over label-linear) or Ir/item ≥ label-mlp16.
```
```python
"""label-rff: Random Fourier Features (RBF kernel approx) + linear head.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x52464631, d, M, C, gamma; float32 W_rff[M][d], b_rff[M], W_lin[M][C], b_lin[C]."""
import json, math, struct, sys

import numpy as np

M = 128          # number of random Fourier features
GAMMA = 0.05     # RBF bandwidth parameter (1/(2*sigma^2))
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
    C = card["classes"]
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    # Standardize features (per-pixel z-score) using training stats
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_std = ((X - mu) / sd).astype(np.float32)

    # Sample random Fourier features: W ~ N(0, gamma*I), b ~ Uniform(0, 2pi)
    W_rff = (rng.normal(0, math.sqrt(GAMMA), (M, d))).astype(np.float32)
    b_rff = (rng.uniform(0, 2 * math.pi, M)).astype(np.float32)

    # Compute RFF map: phi(x) = sqrt(2/M) * cos(W x + b)
    Z = np.cos(X_std @ W_rff.T + b_rff) * math.sqrt(2.0 / M)   # (n, M)

    # Linear head on RFF features (same optimizer as label-linear)
    W_lin = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = Z.shape[0]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W_lin + b_lin, yy)
            grads = [z.T @ g + L2 * W_lin, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights: magic, d, M, C, gamma; then W_rff, b_rff, mu, sd, W_lin, b_lin
    # All float32 little-endian.
    out = struct.pack("<5i1f", 0x52464631, d, M, C, 0, GAMMA)  # 5th int reserved=0
    out += W_rff.astype("<f4").tobytes()
    out += b_rff.astype("<f4").tobytes()
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 pixels). Output: 1 byte class (argmax).
 * Weights layout (little-endian):
 *   int32 magic=0x52464631, d, M, C, reserved; float32 gamma;
 *   float32 W_rff[M][d], b_rff[M], mu[d], sd[d], W_lin[M][C], b_lin[C].
 * No allocations per item; single pass, O(M*d + M*C) flops. */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 24 || hdr[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], M = hdr[2], C = hdr[3];
    float gamma = *(float *)(hdr + 5);
    if (C < 1 || C > 256 || M < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long need = 24 + (long)M * d * 4 + (long)M * 4 + (long)d * 4 + (long)d * 4 + (long)M * C * 4 + (long)C * 4;
    if (nw != need) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, need); return 3; }

    const float *p = (const float *)(wb + 24);
    const float *W_rff = p; p += (long)M * d;
    const float *b_rff = p; p += M;
    const float *mu = p; p += d;
    const float *sd = p; p += d;
    const float *W_lin = p; p += (long)M * C;
    const float *b_lin = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rff = malloc(sizeof(float) * M);
    float *scores = malloc(sizeof(float) * C);
    float *x_std = malloc(sizeof(float) * d);
    const float scale = sqrtf(2.0f / M);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Standardize: (x - mu) / sd
        for (int j = 0; j < d; j++) x_std[j] = (xi[j] - mu[j]) / sd[j];

        // Random Fourier features: cos(W_rff * x_std + b_rff) * scale
        for (int m = 0; m < M; m++) {
            float sum = b_rff[m];
            const float *Wm = W_rff + (long)m * d;
            for (int j = 0; j < d; j++) sum += Wm[j] * x_std[j];
            rff[m] = cosf(sum) * scale;
        }

        // Linear head: scores = rff @ W_lin + b_lin
        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int m = 0; m < M; m++) {
            float rm = rff[m];
            const float *Wmc = W_lin + (long)m * C;
            for (int c = 0; c < C; c++) scores[c] += rm * Wmc[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```