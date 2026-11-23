```hypothesis
Family: label-cnn-tiny (tiny convolutional network). The 8x8 digit images have local spatial structure; a small CNN with 4 3x3 filters, global average pooling, and a linear layer should capture this better than a linear model. It has only ~90 parameters, so it trains fast and infers with ~2k multiply-adds per item. If dev accuracy is not above label-linear (0.9613) or CPU cost exceeds label-linear significantly, the hypothesis fails.
```

```python
"""Tiny CNN for 8x8 optdigits: 4 3x3 filters, ReLU, global avg pool, linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
N_FILTERS = 4
FILTER_SIZE = 3  # 3x3
POOL_SIZE = 8    # global average pooling over 8x8
IN_H, IN_W, IN_C = 8, 8, 1
OUT_H, OUT_W = IN_H, IN_W  # padding=1
CLASSES = 10

rng = np.random.default_rng(SEED)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def im2col(x, kh, kw, pad=1):
    """x: (N, H, W, C) -> (N, OH*OW, kh*kw*C)"""
    N, H, W, C = x.shape
    OH, OW = H + 2 * pad - kh + 1, W + 2 * pad - kw + 1
    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
    col = np.zeros((N, OH * OW, kh * kw * C), dtype=np.float32)
    for i in range(OH):
        for j in range(OW):
            patch = x_pad[:, i:i + kh, j:j + kw, :].reshape(N, -1)
            col[:, i * OW + j, :] = patch
    return col, OH, OW


def softmax_cross_entropy(logits, y):
    """logits: (N, C), y: (N,) int. Returns loss, grad_logits (N, C)"""
    N = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(N), y] -= 1.0
    grad /= N
    return loss, grad.astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    # card["input"]["bytes"] == 64, card["classes"] == 10
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Normalize to [0,1] (max pixel value is 16)
    X = (X_bytes.astype(np.float32) / 16.0).reshape(n, IN_H, IN_W, IN_C)

    # Parameters
    # conv: (N_FILTERS, kh*kw*IN_C)
    conv_w = (rng.uniform(-1, 1, (N_FILTERS, FILTER_SIZE * FILTER_SIZE * IN_C)) *
              math.sqrt(6.0 / (FILTER_SIZE * FILTER_SIZE * IN_C))).astype(np.float32)
    conv_b = np.zeros(N_FILTERS, np.float32)
    # fc: (N_FILTERS, CLASSES)
    fc_w = (rng.uniform(-1, 1, (N_FILTERS, CLASSES)) *
            math.sqrt(6.0 / N_FILTERS)).astype(np.float32)
    fc_b = np.zeros(CLASSES, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Precompute im2col indices? We'll recompute each batch (small).
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(idx)

            # Forward
            col, OH, OW = im2col(xb, FILTER_SIZE, FILTER_SIZE)  # (bsz, 64, 9)
            # conv: (bsz, 64, 9) @ (9, N_FILTERS) -> (bsz, 64, N_FILTERS)
            conv_out = np.matmul(col, conv_w.T) + conv_b  # (bsz, 64, N_FILTERS)
            relu_out = np.maximum(conv_out, 0.0)
            # global average pooling: mean over 64 positions -> (bsz, N_FILTERS)
            pooled = relu_out.mean(axis=1)
            # fc
            logits = np.matmul(pooled, fc_w) + fc_b  # (bsz, CLASSES)

            # Loss & grad
            loss, grad_logits = softmax_cross_entropy(logits, yb)

            # Backward
            # fc
            grad_fc_w = np.matmul(pooled.T, grad_logits)  # (N_FILTERS, CLASSES)
            grad_fc_b = grad_logits.sum(axis=0)
            grad_pooled = np.matmul(grad_logits, fc_w.T)  # (bsz, N_FILTERS)

            # pooling: grad_pooled is mean over 64, so each position gets grad_pooled / 64
            grad_relu = grad_pooled[:, None, :] / 64.0  # (bsz, 1, N_FILTERS) broadcast to (bsz, 64, N_FILTERS)
            # ReLU
            grad_conv = grad_relu * (conv_out > 0.0)

            # conv
            grad_conv_w = np.matmul(col.transpose(0, 2, 1), grad_conv).sum(axis=0)  # (9, N_FILTERS)
            grad_conv_b = grad_conv.sum(axis=(0, 1))

            grads = [grad_conv_w.T, grad_conv_b, grad_fc_w, grad_fc_b]

            # L2 regularization
            for i, p in enumerate(params):
                grads[i] += L2 * p

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights: magic, n_filters, filter_size, classes, in_h, in_w, in_c
    header = struct.pack("<7i", 0x434E4E31, N_FILTERS, FILTER_SIZE, CLASSES, IN_H, IN_W, IN_C)
    # Order: conv_w (N_FILTERS, 9), conv_b (N_FILTERS), fc_w (N_FILTERS, CLASSES), fc_b (CLASSES)
    body = (conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            fc_w.astype("<f4").tobytes() +
            fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Tiny CNN inference for 8x8 optdigits: 4 3x3 filters, ReLU, global avg pool, linear.
 * ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixel counts 0..16). Output: one byte class index. */
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
    if (nw < 28 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_filters = hd[1], filter_size = hd[2], classes = hd[3];
    int in_h = hd[4], in_w = hd[5], in_c = hd[6];
    if (n_filters != 4 || filter_size != 3 || classes != 10 || in_h != 8 || in_w != 8 || in_c != 1) {
        fprintf(stderr, "unexpected architecture\n"); return 3;
    }
    int kh = filter_size, kw = filter_size;
    int oh = in_h, ow = in_w;  // padding=1
    long expected = 28 + 4L * (n_filters * kh * kw * in_c + n_filters + n_filters * classes + classes);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 7);
    const float *conv_w = p; p += n_filters * kh * kw * in_c;
    const float *conv_b = p; p += n_filters;
    const float *fc_w = p; p += n_filters * classes;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize input to [0,1] float
        float input[8][8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                input[r][c] = xi[r * 8 + c] / 16.0f;
            }
        }

        // Convolution + ReLU
        float conv_out[4][8][8];
        for (int f = 0; f < n_filters; f++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = conv_b[f];
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int rr = r + dr, cc = c + dc;
                            if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
                                int w_idx = f * 9 + (dr + 1) * 3 + (dc + 1);
                                sum += input[rr][cc] * conv_w[w_idx];
                            }
                        }
                    }
                    conv_out[f][r][c] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // Global average pooling
        float pooled[4] = {0};
        for (int f = 0; f < n_filters; f++) {
            float sum = 0.0f;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    sum += conv_out[f][r][c];
            pooled[f] = sum / 64.0f;
        }

        // Linear layer
        float scores[10];
        for (int c = 0; c < classes; c++) {
            float s = fc_b[c];
            for (int f = 0; f < n_filters; f++)
                s += pooled[f] * fc_w[f * classes + c];
            scores[c] = s;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < classes; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }
    return 0;
}
```