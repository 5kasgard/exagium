```hypothesis
Family: label-conv. A tiny CNN: one 3×3 conv layer (8 filters, pad=1), ReLU, global average pooling, then linear to 10 classes. It exploits the 2D spatial layout of the 8×8 digits, which linear models ignore. With ~170 weights plus per-pixel normalization, it should beat label-linear's 96.1% dev accuracy while keeping inference cost comparable (~5k multiply-adds). Failure if dev accuracy ≤96.5% or instructions/item >6000.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
NUM_FILTERS = 8
FILTER_SIZE = 3
SEED = 1
LR = 1e-3
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


def im2col(x, k, pad=1, stride=1):
    """x: (N, C, H, W) -> (N, C*k*k, H_out, W_out)"""
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - k) // stride + 1
    W_out = (W + 2 * pad - k) // stride + 1
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    strides = x_pad.strides
    shape = (N, C, H_out, W_out, k, k)
    new_strides = (strides[0], strides[1], strides[2] * stride, strides[3] * stride, strides[2], strides[3])
    windows = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=new_strides)
    return windows.reshape(N, C * k * k, H_out, W_out)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 64
    C = card["classes"]                 # 10
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Per-pixel normalization (mean, std) on training data
    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    X_norm = X_norm.reshape(n, 1, 8, 8).astype(np.float32)

    rng = np.random.default_rng(SEED)

    # Initialize weights
    # conv: (F, C_in, K, K) -> (8, 1, 3, 3)
    fan_in = 1 * FILTER_SIZE * FILTER_SIZE
    conv_w = (rng.uniform(-1, 1, (NUM_FILTERS, 1, FILTER_SIZE, FILTER_SIZE)) *
              math.sqrt(6.0 / fan_in)).astype(np.float32)
    conv_b = np.zeros(NUM_FILTERS, np.float32)
    # linear: (F, C)
    fan_in_lin = NUM_FILTERS
    lin_w = (rng.uniform(-1, 1, (NUM_FILTERS, C)) *
             math.sqrt(6.0 / fan_in_lin)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_norm[idx]
            yy = y[idx]
            m = len(yy)

            # Forward
            col = im2col(x, FILTER_SIZE, pad=1, stride=1)          # (m, 9, 8, 8)
            col_t = col.transpose(0, 2, 3, 1).reshape(-1, 9)       # (m*64, 9)
            conv_w_flat = conv_w.reshape(NUM_FILTERS, 9).T         # (9, F)
            conv_out = col_t @ conv_w_flat + conv_b                # (m*64, F)
            conv_out = conv_out.reshape(m, 8, 8, NUM_FILTERS).transpose(0, 3, 1, 2)  # (m, F, 8, 8)
            relu_out = np.maximum(conv_out, 0.0)
            # Global average pooling
            gap = relu_out.mean(axis=(2, 3))                       # (m, F)
            logits = gap @ lin_w + lin_b                           # (m, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_logits = probs / m                                 # (m, C)

            # Backward linear
            grad_lin_w = gap.T @ grad_logits + L2 * lin_w           # (F, C)
            grad_lin_b = grad_logits.sum(axis=0)                    # (C)

            # Backward gap
            grad_gap = grad_logits @ lin_w.T                        # (m, F)
            grad_relu = grad_gap[:, :, None, None] / 64.0           # (m, F, 1, 1)
            grad_relu = np.broadcast_to(grad_relu, relu_out.shape)  # (m, F, 8, 8)
            grad_relu[relu_out <= 0] = 0.0

            # Backward conv
            grad_relu_flat = grad_relu.transpose(0, 2, 3, 1).reshape(-1, NUM_FILTERS)  # (m*64, F)
            grad_conv_w_flat = col_t.T @ grad_relu_flat + L2 * conv_w_flat              # (9, F)
            grad_conv_b = grad_relu_flat.sum(axis=0)                                      # (F)

            grads = [grad_conv_w_flat.T.reshape(NUM_FILTERS, 1, FILTER_SIZE, FILTER_SIZE),
                     grad_conv_b, grad_lin_w, grad_lin_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                m_hat = m1 / (1 - 0.9 ** t)
                v_hat = v1 / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic 0x434E4E31 ("CNN1"), d, F, K, C
    header = struct.pack("<5i", 0x434E4E31, d, NUM_FILTERS, FILTER_SIZE, C)
    # mean[64], std[64] as float32
    mean_std = np.stack([mean.astype(np.float32), std.astype(np.float32)], axis=1).flatten()
    # conv_w (F,1,K,K), conv_b (F), lin_w (F,C), lin_b (C)
    body = (mean_std.tobytes() +
            conv_w.astype("<f4").tobytes() + conv_b.astype("<f4").tobytes() +
            lin_w.astype("<f4").tobytes() + lin_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


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
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], F = hd[2], K = hd[3], C = hd[4];
    if (d != 64 || K != 3 || F != 8 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }

    float *mean = (float *)(hd + 5);
    float *std  = mean + 64;
    float *conv_w = std + 64;
    float *conv_b = conv_w + F * 1 * K * K;           // 8*9 = 72
    float *lin_w  = conv_b + F;                       // 8
    float *lin_b  = lin_w  + F * C;                   // 8*10 = 80

    long expected = 20 + (64+64+72+8+80+10)*4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_img = malloc(64 * sizeof(float));
    float *col = malloc(64 * 9 * sizeof(float));      // im2col buffer for one image
    float *conv_out = malloc(8 * 8 * 8 * sizeof(float)); // F*H*W
    float *gap = malloc(8 * sizeof(float));
    float *logits = malloc(10 * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize
        for (int j = 0; j < 64; j++) in_img[j] = (xi[j] - mean[j]) / std[j];

        // im2col with pad=1, stride=1 on 8x8 -> 64 patches of 9
        for (int py = 0; py < 8; py++) {
            for (int px = 0; px < 8; px++) {
                float *dst = col + (py * 8 + px) * 9;
                int idx = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = py + dy, xx = px + dx;
                        dst[idx++] = (yy >= 0 && yy < 8 && xx >= 0 && xx < 8) ? in_img[yy * 8 + xx] : 0.0f;
                    }
                }
            }
        }

        // Convolution: col (64,9) * conv_w (9,8) -> conv_out (64,8) then add bias
        for (int f = 0; f < F; f++) {
            const float *wf = conv_w + f * 9;
            for (int p = 0; p < 64; p++) {
                const float *cp = col + p * 9;
                float sum = conv_b[f];
                for (int kk = 0; kk < 9; kk++) sum += cp[kk] * wf[kk];
                conv_out[f * 64 + p] = sum > 0.0f ? sum : 0.0f; // ReLU
            }
        }

        // Global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int p = 0; p < 64; p++) sum += conv_out[f * 64 + p];
            gap[f] = sum / 64.0f;
        }

        // Linear
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            for (int f = 0; f < F; f++) sum += gap[f] * lin_w[f * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```