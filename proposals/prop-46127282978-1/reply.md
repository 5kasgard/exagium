```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).  
Why: Optdigits pixels exhibit strong local pairwise interactions (strokes, curves) that a linear model on raw pixels cannot capture. Expanding each normalized 64-D input to 2144 features (linear, quadratic, and all cross terms) lets a linear classifier model these interactions while keeping inference a single matrix-vector product.  
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 2× label-linear.
```

```python
"""label-poly2: polynomial degree-2 features + multinomial logistic regression.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-2, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"
D_IN = 64
D_POLY = 64 + 64 + 64 * 63 // 2  # 2144


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def poly_expand(Z):
    """Z: (n, 64) float32 -> (n, 2144) float32"""
    n = Z.shape[0]
    Phi = np.empty((n, D_POLY), dtype=np.float32)
    Phi[:, :64] = Z
    Phi[:, 64:128] = Z * Z
    idx = 128
    for i in range(64):
        for j in range(i + 1, 64):
            Phi[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    return Phi


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Normalize per pixel (ordinal)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)

    # Polynomial expansion
    Phi = poly_expand(Z)  # (n, D_POLY)

    rng = np.random.default_rng(SEED)
    # Xavier init for linear layer
    scale = math.sqrt(6.0 / (D_POLY + C))
    W = (rng.uniform(-1, 1, (C, D_POLY)) * scale).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = Phi[idx], y[idx]
            # forward
            logits = Xb @ W.T + b  # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad = probs / len(yb)  # (B, C)
            # gradients
            gW = grad.T @ Xb + L2 * W
            gb = grad.sum(axis=0)
            # Adam update
            t += 1
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)

    # Pack weights: magic, d_in, D_poly, C, mean[64], std[64], W[C][D_POLY], b[C]
    header = struct.pack("<4i", MAGIC, D_IN, D_POLY, C)
    body = mean.tobytes() + std.tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output record: 1 byte (class 0..9). */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || D != 2144 || C != 10) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 64L*4 + 64L*4 + (long)C*D*4 + (long)C*4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + 64;
    const float *W    = std + 64;
    const float *b    = W + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float z[64], score[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        for (int j = 0; j < 64; j++) z[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) score[c] = b[c];

        const float *w = W;
        for (int j = 0; j < 64; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) score[c] += w[c*D + j] * zj;
        }
        for (int j = 0; j < 64; j++) {
            float zj = z[j] * z[j];
            for (int c = 0; c < C; c++) score[c] += w[c*D + 64 + j] * zj;
        }
        long base = 128;
        for (int i1 = 0; i1 < 63; i1++) {
            for (int i2 = i1 + 1; i2 < 64; i2++) {
                float prod = z[i1] * z[i2];
                for (int c = 0; c < C; c++) score[c] += w[c*D + base] * prod;
                base++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```