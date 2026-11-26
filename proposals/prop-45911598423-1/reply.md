```hypothesis
Family: label-rff (Random Fourier Features + linear classifier).
Why: The 8x8 optdigits pixels have non-linear structure that a linear model on z-scored pixels (label-linear, 96.1%) cannot capture. An RBF kernel would help, but exact kernel methods are too slow for inference. Random Fourier Features approximate an RBF kernel with a small, fixed set of cosine features (M=64), then a linear layer on top. This adds non-linearity with only ~5k FLOPs/item (vs ~640 for linear) and ~20 KiB weights. The fixed random projection is generated deterministically from a seed, so training stays reproducible. If dev accuracy does not exceed label-linear, the task is essentially linear or M=64 is insufficient.
```

```python
"""label-rff: Random Fourier Features (RBF kernel approx) + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1 for Adam, seed 12345 for RFF), single thread, stdlib+numpy.
Weights format (little-endian):
  int32 magic=0x52464631, d=64, M=64, C=10
  float32 gamma
  float32 mean[d], std[d]
  float32 W_rff[M][d], b_rff[M]
  float32 W_lin[M][C], b_lin[C]
"""
import json, math, struct, sys

import numpy as np

SEED_ADAM = 1
SEED_RFF = 12345
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
M = 64          # number of random Fourier features
GAMMA = 1.0 / 64.0  # RBF kernel parameter: exp(-gamma * ||x-y||^2)
MAGIC = 0x52464631  # 'RFF1'

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
    assert d == 64 and C == 10

    # Normalize per pixel (z-score on training set)
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xn = (Xf - mean) / std   # (n, d) float64

    # Deterministic RFF projection
    rng_rff = np.random.default_rng(SEED_RFF)
    W_rff = rng_rff.normal(0.0, math.sqrt(GAMMA), size=(M, d)).astype(np.float32)  # (M, d)
    b_rff = rng_rff.uniform(0.0, 2 * math.pi, size=M).astype(np.float32)           # (M,)

    # Compute features for all training data: sqrt(2/M) * cos(Xn @ W_rff.T + b_rff)
    scale = math.sqrt(2.0 / M)
    Z = np.cos(Xn.astype(np.float32) @ W_rff.T + b_rff) * scale   # (n, M) float32

    # Multinomial logistic regression on Z with Adam
    rng = np.random.default_rng(SEED_ADAM)
    W_lin = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W_lin + b_lin
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W_lin
            grad_b = g.sum(0)
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights.bin
    out = bytearray()
    out += struct.pack("<4if", MAGIC, d, M, C, GAMMA)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_rff.astype("<f4").tobytes()
    out += b_rff.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output record: 1 byte (class 0..9).
 * Weights layout (little-endian):
 *   int32 magic=0x52464631, d=64, M=64, C=10
 *   float32 gamma
 *   float32 mean[d], std[d]
 *   float32 W_rff[M][d], b_rff[M]
 *   float32 W_lin[M][C], b_lin[C]
 */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52464631) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (d != 64 || M != 64 || C != 10) { fprintf(stderr, "bad dims\n"); return 3; }

    float *fp = (float *)(hd + 4);
    float gamma = fp[0];
    float *mean = fp + 1;
    float *std  = mean + d;
    float *W_rff = std + d;                    // M * d
    float *b_rff = W_rff + (long)M * d;        // M
    float *W_lin = b_rff + M;                  // M * C
    float *b_lin = W_lin + (long)M * C;        // C

    long expected = 20 + 4L * (1 + d + d + (long)M * d + M + (long)M * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    float *rff  = malloc(sizeof(float) * M);
    float *sc   = malloc(sizeof(float) * C);
    const float scale = sqrtf(2.0f / M);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            norm[j] = (xi[j] - mean[j]) / std[j];
        }

        // Random Fourier features: cos(W_rff * norm + b_rff) * scale
        for (int m = 0; m < M; m++) {
            float sum = b_rff[m];
            const float *w = W_rff + (long)m * d;
            for (int j = 0; j < d; j++) sum += w[j] * norm[j];
            rff[m] = cosf(sum) * scale;
        }

        // Linear layer
        for (int c = 0; c < C; c++) sc[c] = b_lin[c];
        for (int m = 0; m < M; m++) {
            float rm = rff[m];
            const float *w = W_lin + (long)m * C;
            for (int c = 0; c < C; c++) sc[c] += rm * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```