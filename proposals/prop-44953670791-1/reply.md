```hypothesis
Family: label-cnn (tiny convolutional network). The 8x8 digit images have local spatial structure; a 3x3 convolution with 8 filters, ReLU, global max pooling, and a linear layer should capture stroke patterns better than a linear model. It adds minimal inference cost (few hundred multiply-adds per item). If dev accuracy is below 0.96, the hypothesis fails.
```

```python
"""label-cnn: tiny CNN for 8x8 ordinal images (optdigits).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x434E4E31, in_h, in_w, in_c, kernel, filters, classes;
float32 conv_w[filters][in_c][kernel][kernel], conv_b[filters],
fc_w[classes][filters], fc_b[classes] (all little-endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS = 1e-3, 1e-4, 64, 100
FILTERS, KERNEL = 8, 3


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class CNN:
    def __init__(self, in_h, in_w, in_c, kernel, filters, classes, seed):
        rng = np.random.default_rng(seed)
        self.in_h, self.in_w, self.in_c = in_h, in_w, in_c
        self.kernel = kernel
        self.filters = filters
        self.classes = classes
        self.out_h = in_h - kernel + 1
        self.out_w = in_w - kernel + 1
        fan_in = in_c * kernel * kernel
        self.W_conv = rng.normal(0, math.sqrt(2.0 / fan_in),
                                 size=(filters, in_c, kernel, kernel)).astype(np.float32)
        self.b_conv = np.zeros(filters, np.float32)
        self.W_fc = rng.normal(0, math.sqrt(2.0 / (filters + classes)),
                               size=(classes, filters)).astype(np.float32)
        self.b_fc = np.zeros(classes, np.float32)
        # buffers for backward
        self.x = self.patches = self.conv_pre = self.relu = self.max_idx = self.pooled = None

    def im2col(self, x):
        N, C, H, W = x.shape
        k = self.kernel
        oh, ow = H - k + 1, W - k + 1
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(x, (k, k), axis=(2, 3))          # (N, C, oh, ow, k, k)
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * oh * ow, C * k * k)
        return patches, (N, oh, ow)

    def forward(self, x):
        self.x = x
        N = x.shape[0]
        self.patches, (N, oh, ow) = self.im2col(x)
        Wc = self.W_conv.reshape(self.filters, -1)                     # (filters, C*k*k)
        conv = self.patches @ Wc.T + self.b_conv                       # (N*oh*ow, filters)
        self.conv_pre = conv.reshape(N, oh, ow, self.filters).transpose(0, 3, 1, 2)  # (N, F, oh, ow)
        self.relu = np.maximum(self.conv_pre, 0)
        relu_flat = self.relu.reshape(N, self.filters, -1)
        max_vals = np.max(relu_flat, axis=2)
        max_idx = np.argmax(relu_flat, axis=2)
        self.max_idx = np.stack([max_idx // ow, max_idx % ow], axis=2)  # (N, F, 2)
        self.pooled = max_vals                                          # (N, F)
        logits = self.pooled @ self.W_fc.T + self.b_fc                 # (N, C)
        return logits

    def backward(self, d_logits):
        N = d_logits.shape[0]
        oh, ow = self.out_h, self.out_w
        # fc grads
        self.dW_fc = d_logits.T @ self.pooled
        self.db_fc = d_logits.sum(axis=0)
        # pooled grad
        d_pooled = d_logits @ self.W_fc                                 # (N, F)
        # maxpool grad
        d_relu = np.zeros_like(self.relu)
        for n in range(N):
            for f in range(self.filters):
                h, w = self.max_idx[n, f]
                d_relu[n, f, h, w] = d_pooled[n, f]
        # ReLU grad
        d_conv_pre = d_relu * (self.conv_pre > 0)
        # conv grads
        d_conv_flat = d_conv_pre.transpose(0, 2, 3, 1).reshape(N * oh * ow, self.filters)
        self.dW_conv = (d_conv_flat.T @ self.patches).reshape(self.filters, self.in_c, self.kernel, self.kernel)
        self.db_conv = d_conv_flat.sum(axis=0)
        # input grad not needed

    def params_and_grads(self):
        return ([self.W_conv, self.b_conv, self.W_fc, self.b_fc],
                [self.dW_conv, self.db_conv, self.dW_fc, self.db_fc])


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64 and card["classes"] == 10
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, 64).astype(np.float32) / 16.0
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    n = X.shape[0]
    X = X.reshape(n, 1, 8, 8)

    model = CNN(8, 8, 1, KERNEL, FILTERS, 10, SEED)
    rng = np.random.default_rng(SEED)
    params, _ = model.params_and_grads()
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits = model.forward(xb)
            # softmax cross-entropy gradient
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            model.backward(probs.astype(np.float32))
            _, grads = model.params_and_grads()
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g += L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export weights
    magic = 0x434E4E31  # 'CNN1'
    hdr = struct.pack("<7i", magic, 8, 8, 1, KERNEL, FILTERS, 10)
    w_conv = model.W_conv.astype("<f4").tobytes()
    b_conv = model.b_conv.astype("<f4").tobytes()
    w_fc = model.W_fc.astype("<f4").tobytes()
    b_fc = model.b_fc.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + w_conv + b_conv + w_fc + b_fc)


if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N */
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
    int in_h = hd[1], in_w = hd[2], in_c = hd[3], kernel = hd[4], filters = hd[5], classes = hd[6];
    if (in_h != 8 || in_w != 8 || in_c != 1 || kernel != 3 || filters != 8 || classes != 10) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    int out_h = in_h - kernel + 1;  // 6
    int out_w = in_w - kernel + 1;  // 6
    size_t exp_floats = (size_t)filters * in_c * kernel * kernel + filters + (size_t)classes * filters + classes;
    if (nw != 28 + (long)exp_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 7);
    const float *W_conv = p; p += filters * in_c * kernel * kernel;
    const float *b_conv = p; p += filters;
    const float *W_fc = p; p += (size_t)classes * filters;
    const float *b_fc = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float input[8][8];
    float conv_out[8][6][6];
    float pooled[8];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                input[r][c] = xi[r * 8 + c] / 16.0f;

        // convolution + bias
        for (int f = 0; f < filters; f++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = b_conv[f];
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            int ih = oh + kh, iw = ow + kw;
                            sum += input[ih][iw] * W_conv[((f * in_c) * kernel + kh) * kernel + kw];
                        }
                    }
                    conv_out[f][oh][ow] = sum;
                }
            }
        }
        // ReLU + global max pooling
        for (int f = 0; f < filters; f++) {
            float maxv = -1e30f;
            for (int oh = 0; oh < out_h; oh++)
                for (int ow = 0; ow < out_w; ow++)
                    if (conv_out[f][oh][ow] > maxv) maxv = conv_out[f][oh][ow];
            pooled[f] = maxv > 0 ? maxv : 0;
        }
        // fully connected
        for (int c = 0; c < classes; c++) {
            float sum = b_fc[c];
            for (int f = 0; f < filters; f++)
                sum += pooled[f] * W_fc[c * filters + f];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < classes; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```