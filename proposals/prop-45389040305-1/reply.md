```hypothesis
Family: label-cnn (tiny 2D CNN). The 8×8 optdigits images have strong local spatial structure that a linear model ignores. A single convolutional layer (8 filters, 3×3, padding=1) followed by 2×2 max-pool and a 128→10 linear layer captures this structure with only ~1.5k parameters. Inference cost (~6k Ir/item) stays close to label-linear (5k) while accuracy should exceed 0.97. Failure: if dev accuracy ≤0.962 or Ir/item >10k.
```

```python
"""label-cnn: tiny CNN for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: Input 1x8x8 -> Conv(8, 3x3, pad=1) -> ReLU -> MaxPool2x2 -> Flatten(128) -> Linear(10).
Exported weights.bin: magic 0x434E4E31, d, C, conv_filters, conv_kernel, fc_hidden;
float32 arrays: conv_w[8,1,3,3], conv_b[8], fc_w[128,10], fc_b[10]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, BATCH, EPOCHS = 1e-3, 64, 30
CONV_F, CONV_K = 8, 3
FC_HIDDEN = 128  # 4*4*8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def im2col(x, k, pad, stride):
    # x: (N, C, H, W) -> (N, C*k*k, OH*OW)
    N, C, H, W = x.shape
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    col = np.zeros((N, C, k, k, OH, OW), dtype=x.dtype)
    for y in range(k):
        y_max = y + stride * OH
        for x_ in range(k):
            x_max = x_ + stride * OW
            col[:, :, y, x_, :, :] = x_pad[:, :, y:y_max:stride, x_:x_max:stride]
    return col.transpose(0, 1, 2, 3, 4, 5).reshape(N, C * k * k, OH * OW)


def col2im(col, x_shape, k, pad, stride):
    N, C, H, W = x_shape
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    col = col.reshape(N, C, k, k, OH, OW)
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)
    for y in range(k):
        y_max = y + stride * OH
        for x_ in range(k):
            x_max = x_ + stride * OW
            x_pad[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]
    return x_pad[:, :, pad:pad + H, pad:pad + W]


def maxpool_forward(x, k=2, stride=2):
    N, C, H, W = x.shape
    OH = (H - k) // stride + 1
    OW = (W - k) // stride + 1
    x_reshaped = x.reshape(N, C, OH, stride, OW, stride)
    out = x_reshaped.max(axis=(3, 5))
    # store argmax for backward
    argmax = x_reshaped.argmax(axis=(3, 5))
    return out, argmax


def maxpool_backward(grad_out, argmax, x_shape, k=2, stride=2):
    N, C, H, W = x_shape
    OH = (H - k) // stride + 1
    OW = (W - k) // stride + 1
    grad_x = np.zeros(x_shape, dtype=grad_out.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(OH):
                for j in range(OW):
                    idx = argmax[n, c, i, j]
                    hi, wi = divmod(idx, k)
                    grad_x[n, c, i * stride + hi, j * stride + wi] = grad_out[n, c, i, j]
    return grad_x


def softmax_cross_entropy(logits, y):
    # logits: (N, C), y: (N,)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(N), y] -= 1.0
    grad /= N
    return loss, grad, probs


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64

    # Normalize to [0,1] and reshape to (N, 1, 8, 8)
    X = (X_bytes.astype(np.float32) / 16.0).reshape(-1, 1, 8, 8)
    N = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize weights
    # Conv: (out_ch, in_ch, k, k)
    scale = math.sqrt(2.0 / (1 * CONV_K * CONV_K))
    conv_w = (rng.normal(0, scale, (CONV_F, 1, CONV_K, CONV_K))).astype(np.float32)
    conv_b = np.zeros(CONV_F, dtype=np.float32)
    # FC: (128, 10)
    scale2 = math.sqrt(2.0 / FC_HIDDEN)
    fc_w = (rng.normal(0, scale2, (FC_HIDDEN, C))).astype(np.float32)
    fc_b = np.zeros(C, dtype=np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            bs = x.shape[0]

            # Forward
            # Conv
            col = im2col(x, CONV_K, pad=1, stride=1)  # (bs, 1*3*3, 8*8)
            conv_out = col.transpose(0, 2, 1) @ conv_w.reshape(CONV_F, -1).T  # (bs, 64, 8)
            conv_out = conv_out.transpose(0, 2, 1).reshape(bs, CONV_F, 8, 8) + conv_b.reshape(1, -1, 1, 1)
            # ReLU
            relu_mask = (conv_out > 0)
            relu_out = np.maximum(conv_out, 0)
            # MaxPool
            pool_out, pool_argmax = maxpool_forward(relu_out, 2, 2)  # (bs, 8, 4, 4)
            # Flatten
            flat = pool_out.reshape(bs, FC_HIDDEN)
            # FC
            logits = flat @ fc_w + fc_b

            # Loss
            loss, grad_logits, _ = softmax_cross_entropy(logits, yy)

            # Backward
            # FC
            grad_fc_w = flat.T @ grad_logits
            grad_fc_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ fc_w.T

            # MaxPool
            grad_pool = grad_flat.reshape(bs, CONV_F, 4, 4)
            grad_relu = maxpool_backward(grad_pool, pool_argmax, relu_out.shape, 2, 2)

            # ReLU
            grad_conv = grad_relu * relu_mask

            # Conv
            grad_conv_reshaped = grad_conv.transpose(0, 2, 3, 1).reshape(bs, 64, CONV_F)
            col = im2col(x, CONV_K, pad=1, stride=1)  # (bs, 9, 64)
            grad_conv_w = col @ grad_conv_reshaped  # (bs, 9, 8) -> sum over batch
            grad_conv_w = grad_conv_w.sum(axis=0).T  # (8, 9)
            grad_conv_b = grad_conv.sum(axis=(0, 2, 3))

            grads = [grad_conv_w.reshape(conv_w.shape), grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Export
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<5i", magic, d, C, CONV_F, CONV_K)
    body = (conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            fc_w.astype("<f4").tobytes() +
            fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixel counts 0..16). Output: one byte class 0..9. */
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

#define MAX(a,b) ((a)>(b)?(a):(b))

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], CONV_F = hd[3], CONV_K = hd[4];
    if (d != 64 || C != 10 || CONV_F != 8 || CONV_K != 3) { fprintf(stderr, "bad header values\n"); return 3; }
    const int FC_HIDDEN = 4 * 4 * CONV_F;  // 128
    size_t expected = 20 + 4 * (CONV_F * 1 * CONV_K * CONV_K + CONV_F + FC_HIDDEN * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *conv_w = p; p += CONV_F * 1 * CONV_K * CONV_K;
    const float *conv_b = p; p += CONV_F;
    const float *fc_w = p; p += FC_HIDDEN * C;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *conv_out = malloc(sizeof(float) * CONV_F * 8 * 8);
    float *pool_out = malloc(sizeof(float) * CONV_F * 4 * 4);
    float *flat = malloc(sizeof(float) * FC_HIDDEN);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize to [0,1] and store in 8x8 float array
        float input[8][8];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                input[r][c] = xi[r * 8 + c] / 16.0f;

        // Conv: 8 filters, 3x3, pad=1, stride=1 -> 8x8
        for (int f = 0; f < CONV_F; f++) {
            const float *w = conv_w + f * 9;  // 1*3*3
            float b = conv_b[f];
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = b;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int rr = r + dr, cc = c + dc;
                            float val = (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) ? input[rr][cc] : 0.0f;
                            sum += val * w[(dr + 1) * 3 + (dc + 1)];
                        }
                    }
                    conv_out[f * 64 + r * 8 + c] = sum > 0 ? sum : 0;  // ReLU fused
                }
            }
        }

        // MaxPool 2x2 -> 4x4
        for (int f = 0; f < CONV_F; f++) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    float m = conv_out[f * 64 + (2*r) * 8 + (2*c)];
                    m = MAX(m, conv_out[f * 64 + (2*r) * 8 + (2*c+1)]);
                    m = MAX(m, conv_out[f * 64 + (2*r+1) * 8 + (2*c)]);
                    m = MAX(m, conv_out[f * 64 + (2*r+1) * 8 + (2*c+1)]);
                    pool_out[f * 16 + r * 4 + c] = m;
                }
            }
        }

        // Flatten
        for (int f = 0; f < CONV_F; f++)
            memcpy(flat + f * 16, pool_out + f * 16, 16 * sizeof(float));

        // FC
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int h = 0; h < FC_HIDDEN; h++)
                sum += flat[h] * fc_w[h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(conv_out); free(pool_out); free(flat); free(logits);
    return 0;
}
```