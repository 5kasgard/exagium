```hypothesis
We propose "label-tinycnn", a tiny convolutional network with one 3×3 convolutional layer (8 filters), ReLU, global average pooling, and a linear classifier. It exploits the 2D spatial structure of the 8×8 optdigits images, which should yield higher accuracy than the linear baseline while keeping inference cost modest (similar to label-mlp16). If dev accuracy does not exceed label-linear (0.9613) or inference cost exceeds label-mlp16 (8685 Ir/item), the hypothesis fails.
```

```python
"""label-tinycnn: one 3x3 conv (8 filters), ReLU, global avg pool, linear.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
H = 8               # number of convolutional filters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col(x, kh, kw):
    """x: (N, 1, H, W) -> (N, H_out*W_out, kh*kw)"""
    N, _, H, W = x.shape
    H_out = H - kh + 1
    W_out = W - kw + 1
    strides = x.strides
    # (N, H_out, W_out, kh, kw)
    shape = (N, H_out, W_out, kh, kw)
    strides = (strides[0], strides[2], strides[3], strides[2], strides[3])
    cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return cols.reshape(N, H_out * W_out, kh * kw)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    assert d == 64

    rng = np.random.default_rng(SEED)

    # Per-pixel normalization (ordinal encoding)
    mean = X_bytes.mean(axis=0).astype(np.float32)
    std = X_bytes.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X = ((X_bytes - mean) / std).astype(np.float32)  # (N, 64)
    X = X.reshape(-1, 1, 8, 8)  # (N, 1, 8, 8)
    n = X.shape[0]

    # Initialize parameters
    # conv_w: (H, 1, 3, 3) -> (H, 9)
    fan_in = 9
    conv_w = (rng.normal(0, math.sqrt(2.0 / fan_in), (H, 9))).astype(np.float32)
    conv_b = np.zeros(H, np.float32)
    # linear_w: (H, C)
    fan_in_lin = H
    linear_w = (rng.normal(0, math.sqrt(2.0 / fan_in_lin), (H, C))).astype(np.float32)
    linear_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, linear_w, linear_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]          # (B, 1, 8, 8)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # im2col: (B, 36, 9)
            cols = im2col(x_batch, 3, 3)
            # conv: (B, 36, H)
            conv_out = cols @ conv_w.T + conv_b  # (B, 36, H)
            relu_out = np.maximum(conv_out, 0)
            # global average pool over 36 positions: (B, H)
            pool_out = relu_out.mean(axis=1)
            # linear: (B, C)
            scores = pool_out @ linear_w + linear_b

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_scores = probs / B  # (B, C)

            # Backward linear
            grad_linear_w = pool_out.T @ grad_scores + L2 * linear_w  # (H, C)
            grad_linear_b = grad_scores.sum(axis=0)                   # (C,)
            grad_pool = grad_scores @ linear_w.T                      # (B, H)

            # Backward pool (average over 36)
            grad_relu = grad_pool[:, None, :] / 36.0  # (B, 1, H) -> broadcast to (B, 36, H)
            grad_relu = np.repeat(grad_relu, 36, axis=1)  # (B, 36, H)
            grad_relu[relu_out == 0] = 0.0

            # Backward conv
            grad_conv_w = grad_relu.transpose(0, 2, 1) @ cols  # (B, H, 36) @ (B, 36, 9) -> sum over B -> (H, 9)
            grad_conv_w = grad_conv_w.sum(axis=0) + L2 * conv_w
            grad_conv_b = grad_relu.sum(axis=(0, 1))

            grads = [grad_conv_w, grad_conv_b, grad_linear_w, grad_linear_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, H, C; mean[d]; std[d]; conv_w[H*9]; conv_b[H]; linear_w[H*C]; linear_b[C]
    magic = 0x434E5631  # 'CNV1'
    header = struct.pack("<4i", magic, d, H, C)
    body = (mean.tobytes() + std.tobytes() +
            conv_w.tobytes() + conv_b.tobytes() +
            linear_w.tobytes() + linear_b.tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-tinycnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8). Output: one byte class (0-9). */
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
    if (nw < 16 || hd[0] != 0x434E5631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 8 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16L + 4L * (d + d + H*9 + H + H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *conv_w = std + d;
    const float *conv_b = conv_w + H * 9;
    const float *linear_w = conv_b + H;
    const float *linear_b = linear_w + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float norm_img[64];
    float pool_sum[8];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int p = 0; p < d; p++) {
            norm_img[p] = (xi[p] - mean[p]) / std[p];
        }

        // Convolution + ReLU + accumulate for global average pooling
        for (int h = 0; h < H; h++) pool_sum[h] = 0.0f;

        for (int h = 0; h < H; h++) {
            const float *w = conv_w + h * 9;
            for (int iy = 0; iy < 6; iy++) {
                for (int ix = 0; ix < 6; ix++) {
                    float sum = conv_b[h];
                    for (int dy = 0; dy < 3; dy++) {
                        for (int dx = 0; dx < 3; dx++) {
                            int py = iy + dy;
                            int px = ix + dx;
                            float val = norm_img[py * 8 + px];
                            sum += val * w[dy * 3 + dx];
                        }
                    }
                    if (sum < 0.0f) sum = 0.0f; // ReLU
                    pool_sum[h] += sum;
                }
            }
        }

        // Global average pooling (divide by 36)
        for (int h = 0; h < H; h++) pool_sum[h] /= 36.0f;

        // Linear layer
        for (int c = 0; c < C; c++) {
            float s = linear_b[c];
            for (int h = 0; h < H; h++) {
                s += pool_sum[h] * linear_w[h * C + c];
            }
            scores[c] = s;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```