```hypothesis
Family: label-rff (Random Fourier Features + linear classifier). The 8x8 optdigits pixels have non-linear boundaries; an RBF kernel typically reaches ~98% on this task. RFF approximates the RBF kernel with an explicit finite-dimensional feature map (cosine projections), allowing a fast linear inference. We fix D=256 features and gamma=1/64 (median heuristic). If dev accuracy does not exceed label-linear (0.9613), the non-linearity is not exploitable with this bandwidth/feature count.
```
```python
"""label-rff: Random Fourier Features for shift-invariant kernel approximation + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x5246464C, int32 d, int32 D, int32 C, float32 gamma,
  float32 w[D][d], float32 b[D], float32 W[D][C], float32 b_lin[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split: last 20% of train.in)
D = 256               # number of random Fourier features
GAMMA = 1.0 / 64.0    # RBF bandwidth
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X_bytes.shape
    C = card["classes"]

    # Normalize pixels to [0,1] (max 16 per card description)
    X = X_bytes.astype(np.float32) / 16.0

    # Generate random Fourier features: w ~ N(0, sqrt(2*gamma)), b ~ U(0, 2pi)
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, math.sqrt(2 * GAMMA), size=(D, d)).astype(np.float32)
    b = rng.uniform(0, 2 * math.pi, size=D).astype(np.float32)

    # Compute feature matrix Z = sqrt(2/D) * cos(X @ w.T + b)
    Z = np.cos(X @ w.T + b) * math.sqrt(2.0 / D)  # (n, D)

    # Linear model on Z with Adam (same as label-linear)
    W_lin = np.zeros((D, C), np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W_lin + b_lin, yy)
            grads = [z.T @ g + L2 * W_lin, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    magic = 0x5246464C  # 'LFFR' in little-endian
    header = struct.pack("<4if", magic, d, D, C, GAMMA)
    body = w.astype("<f4").tobytes() + b.astype("<f4").tobytes() + \
           W_lin.astype("<f4").tobytes() + b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-rff infer: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class (argmax). */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hdr_i = (int32_t *)wb;
    float *hdr_f = (float *)(wb + 16);
    if (hdr_i[0] != 0x5246464C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr_i[1], D = hdr_i[2], C = hdr_i[3];
    float gamma = hdr_f[0];
    (void)gamma; /* unused in inference, kept for validation */

    long expected = 20L + 4L * ((long)D * d + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *w = hdr_f + 1;
    const float *b = w + (long)D * d;
    const float *W = b + D;
    const float *b_lin = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    float scale = sqrtf(2.0f / D);
    const float inv16 = 1.0f / 16.0f;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Compute RFF features: z[j] = cos(w[j]^T x + b[j]) * sqrt(2/D) */
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            const float *wj = w + (long)j * d;
            for (int p = 0; p < d; p++) sum += wj[p] * (xi[p] * inv16);
            z[j] = cosf(sum + b[j]) * scale;
        }

        /* Linear classification */
        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int j = 0; j < D; j++) {
            float zj = z[j];
            const float *Wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += zj * Wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```