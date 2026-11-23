```hypothesis
Family: label-cnn1 (tiny 1-layer CNN). The 8x8 digit images have local spatial structure that a 3x3 convolution can exploit, which linear models ignore. With 8 filters and a small fully-connected layer, parameter count stays ~3k, keeping inference fast. If dev accuracy does not exceed label-linear (0.9613) or latency is higher, the approach fails.
```

```python
"""Tiny CNN for 8x8 optdigits: 8 filters 3x3, ReLU, FC to 10 classes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
H = 8               # number of conv filters
KERNEL = 3          # 3x3
EPOCHS = 50
BATCH = 64
LR = 1e-3
SEED = 1
EPS = 1e-8

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

    rng = np.random.default_rng(SEED)

    # Per-pixel normalization (ordinal)
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0) + EPS
    Xn = (Xf - mean) / std               # (n, 64)
    Xn = Xn.reshape(n, 1, 8, 8)          # (n, 1, 8, 8)

    # Conv output size: 8-3+1 = 6
    OH = OW = 6
    flat_dim = H * OH * OW

    # Initialize weights
    # Conv: (H, 1, 3, 3) -> He init
    fan_in = 1 * KERNEL * KERNEL
    W_conv = rng.normal(0, math.sqrt(2.0 / fan_in), size=(H, 1, KERNEL, KERNEL)).astype(np.float32)
    b_conv = np.zeros(H, np.float32)
    # FC: (flat_dim, C)
    fan_in_fc = flat_dim
    W_fc = rng.normal(0, math.sqrt(2.0 / fan_in_fc), size=(flat_dim, C)).astype(np.float32)
    b_fc = np.zeros(C, np.float32)

    # Adam state
    params = [W_conv, b_conv, W_fc, b_fc]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.999
    t = 0

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = Xn[idx]          # (b, 1, 8, 8)
            yb = y[idx]          # (b,)
            b = x.shape[0]

            # Forward conv
            # im2col: (b, 1, 8, 8) -> (b, OH*OW, 1*KERNEL*KERNEL)
            # We'll do simple loops for clarity (b small)
            conv_out = np.zeros((b, H, OH, OW), np.float32)
            for i in range(OH):
                for j in range(OW):
                    patch = x[:, :, i:i+KERNEL, j:j+KERNEL]   # (b, 1, 3, 3)
                    patch_flat = patch.reshape(b, -1)         # (b, 9)
                    # (b, 9) @ (9, H) -> (b, H)
                    conv_out[:, :, i, j] = patch_flat @ W_conv.reshape(-1, H) + b_conv

            # ReLU
            relu_out = np.maximum(conv_out, 0)

            # Flatten
            flat = relu_out.reshape(b, flat_dim)

            # FC
            logits = flat @ W_fc + b_fc   # (b, C)

            # Softmax + cross-entropy gradient
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(b), yb] -= 1.0
            grad_logits = probs / b       # (b, C)

            # Backprop FC
            grad_W_fc = flat.T @ grad_logits
            grad_b_fc = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ W_fc.T

            # Backprop ReLU
            grad_relu = grad_flat.reshape(b, H, OH, OW) * (relu_out > 0)

            # Backprop conv
            grad_W_conv = np.zeros_like(W_conv)
            grad_b_conv = np.zeros_like(b_conv)
            for i in range(OH):
                for j in range(OW):
                    patch = x[:, :, i:i+KERNEL, j:j+KERNEL].reshape(b, -1)  # (b, 9)
                    g = grad_relu[:, :, i, j]                               # (b, H)
                    grad_W_conv += patch.T @ g
                    grad_b_conv += g.sum(axis=0)
            grad_W_conv = grad_W_conv.reshape(W_conv.shape)

            grads = [grad_W_conv, grad_b_conv, grad_W_fc, grad_b_fc]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Header: magic, d, H, C, kernel
    magic = 0x434E4E31  # 'CNN1' little-endian
    header = struct.pack("<5i", magic, d, H, C, KERNEL)
    # mean, std (d each)
    body = mean.astype("<f4").tobytes() + std.astype("<f4").tobytes()
    # conv weights (H, 1, 3, 3) -> flatten
    body += W_conv.astype("<f4").tobytes()
    body += b_conv.astype("<f4").tobytes()
    # fc weights (flat_dim, C), bias (C)
    body += W_fc.astype("<f4").tobytes()
    body += b_fc.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Tiny CNN inference for 8x8 optdigits: ./infer weights.bin in.rec N
 * Each input record: 64 bytes (uint8 pixels). Output: 1 byte class.
 * Weights format: magic(0x434E4E31), d, H, C, kernel; float32 mean[d], std[d],
 * conv_w[H*kernel*kernel], conv_b[H], fc_w[H*6*6*C], fc_b[C]. */
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

    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], H = hdr[2], C = hdr[3], K = hdr[4];
    if (d != 64 || K != 3 || C != 10 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }

    const int OH = 6, OW = 6;
    const int flat_dim = H * OH * OW;
    long expected = 20L + (long)d * 4 * 2 + (long)H * K * K * 4 + (long)H * 4 +
                    (long)flat_dim * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hdr + 5);
    const float *std  = mean + d;
    const float *conv_w = std + d;
    const float *conv_b = conv_w + (long)H * K * K;
    const float *fc_w = conv_b + H;
    const float *fc_b = fc_w + (long)flat_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_f = malloc(d * sizeof(float));
    float *conv_out = malloc((long)H * OH * OW * sizeof(float));
    float *flat = malloc(flat_dim * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++)
            in_f[j] = (xi[j] - mean[j]) / std[j];

        // Convolution + ReLU
        for (int h = 0; h < H; h++) {
            const float *w = conv_w + h * K * K;
            for (int oy = 0; oy < OH; oy++) {
                for (int ox = 0; ox < OW; ox++) {
                    float sum = conv_b[h];
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int ix = ox + kx;
                            int iy = oy + ky;
                            int idx = iy * 8 + ix;
                            sum += in_f[idx] * w[ky * K + kx];
                        }
                    }
                    conv_out[(h * OH + oy) * OW + ox] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // Flatten (already in conv_out order: H, OH, OW)
        memcpy(flat, conv_out, flat_dim * sizeof(float));

        // FC
        for (int c = 0; c < C; c++) scores[c] = fc_b[c];
        for (int f = 0; f < flat_dim; f++) {
            float val = flat[f];
            if (val == 0.0f) continue;
            const float *w = fc_w + f * C;
            for (int c = 0; c < C; c++) scores[c] += val * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(in_f); free(conv_out); free(flat); free(scores);
    if (x) free(x); free(wb);
    return 0;
}
```