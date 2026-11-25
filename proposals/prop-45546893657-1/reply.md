```hypothesis
We propose label-tiny-cnn: a tiny CNN with one 3x3 convolutional layer (16 filters, padding=1), ReLU, global average pooling, and a linear classifier. It exploits the 8x8 spatial structure of optdigits, which linear models ignore, and weight sharing keeps parameters low (~1k). Training uses Adam (lr=0.01, batch=64, epochs=50) on normalized pixels (0..16 -> /16). Expect dev accuracy >0.965 (beating label-linear 0.9613) with inference cost similar to label-linear due to small model. If dev accuracy <0.96, the spatial inductive bias does not help.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
SEED = 1
LR = 0.01
BATCH = 64
EPOCHS = 50
FILTERS = 16          # number of conv filters
KERNEL = 3
PADDING = 1
L2 = 1e-4
IN_H, IN_W = 8, 8
IN_CH = 1
OUT_CLASSES = 10

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    d = card["input"]["bytes"]          # 64
    C = card["classes"]                 # 10
    assert d == IN_H * IN_W * IN_CH

    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d).astype(np.float32) / 16.0
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    n = X.shape[0]

    # Reshape to NCHW
    X = X.reshape(n, IN_CH, IN_H, IN_W)

    rng = np.random.default_rng(SEED)

    # Initialize weights
    # conv: (F, C, K, K)
    scale = math.sqrt(2.0 / (IN_CH * KERNEL * KERNEL))
    W_conv = rng.normal(0, scale, (FILTERS, IN_CH, KERNEL, KERNEL)).astype(np.float32)
    b_conv = np.zeros(FILTERS, np.float32)
    # linear: (F, C)
    scale2 = math.sqrt(2.0 / FILTERS)
    W_lin = rng.normal(0, scale2, (FILTERS, C)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)

    params = [W_conv, b_conv, W_lin, b_lin]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(x):
        # x: (B, 1, 8, 8)
        B = x.shape[0]
        # conv with padding=1 -> output (B, F, 8, 8)
        out_h = IN_H + 2*PADDING - KERNEL + 1
        out_w = IN_W + 2*PADDING - KERNEL + 1
        assert out_h == IN_H and out_w == IN_W
        # im2col for simplicity (small sizes)
        col = np.zeros((B, IN_CH * KERNEL * KERNEL, out_h * out_w), np.float32)
        for i in range(KERNEL):
            for j in range(KERNEL):
                h_start = i - PADDING
                w_start = j - PADDING
                h_end = h_start + out_h
                w_end = w_start + out_w
                col_idx = (i * KERNEL + j) * IN_CH
                if 0 <= h_start and h_end <= IN_H and 0 <= w_start and w_end <= IN_W:
                    col[:, col_idx:col_idx+IN_CH, :] = x[:, :, h_start:h_end, w_start:w_end].reshape(B, IN_CH, -1)
                else:
                    # padding handled by zero initialization
                    pass
        # W_conv: (F, C*K*K) -> (F, out_h*out_w)
        Wc = W_conv.reshape(FILTERS, -1)
        conv_out = Wc @ col + b_conv[:, None]  # (F, out_h*out_w) per batch? Need batch matmul
        # Actually we need per batch: (B, F, H, W)
        conv_out = conv_out.reshape(B, FILTERS, out_h, out_w)
        # ReLU
        h = np.maximum(conv_out, 0)
        # Global average pooling
        gap = h.mean(axis=(2,3))  # (B, F)
        # Linear
        logits = gap @ W_lin + b_lin  # (B, C)
        return logits, h, gap

    def softmax_cross_entropy(logits, y):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        grad = probs
        grad[np.arange(len(y)), y] -= 1
        grad /= len(y)
        return loss, grad, probs

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            logits, h, gap = forward(xb)
            loss, dlogits, _ = softmax_cross_entropy(logits, yb)

            # Backprop
            # dlogits: (B, C)
            # grad W_lin: (F, C) = gap.T @ dlogits
            dW_lin = gap.T @ dlogits + L2 * W_lin
            db_lin = dlogits.sum(axis=0)
            # dgap: (B, F) = dlogits @ W_lin.T
            dgap = dlogits @ W_lin.T
            # dh: (B, F, H, W) = dgap[:,:,None,None] / (H*W)
            dh = dgap[:, :, None, None] / (IN_H * IN_W)
            # ReLU grad
            dh[h <= 0] = 0
            # dW_conv: (F, C*K*K) = dh_flat @ col_flat.T? Need to compute per batch and sum.
            # Reshape dh to (B, F, H*W)
            dh_flat = dh.reshape(B, FILTERS, -1)  # (B, F, 64)
            # col is (B, C*K*K, H*W)
            # dW_conv = sum_b (dh_flat[b] @ col[b].T) -> (F, C*K*K)
            dW_conv = np.zeros_like(Wc)
            for b in range(B):
                dW_conv += dh_flat[b] @ col[b].T
            dW_conv += L2 * Wc
            db_conv = dh_flat.sum(axis=(0,2))

            grads = [dW_conv.reshape(W_conv.shape), db_conv, dW_lin, db_lin]

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Export weights
    # Format: magic, d, FILTERS, C, then conv weights (F*C*K*K), conv bias (F), linear weights (F*C), linear bias (C)
    magic = 0x544E4301  # 'TCN1'
    out = struct.pack("<4i", magic, d, FILTERS, C)
    out += W_conv.astype("<f4").tobytes()
    out += b_conv.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x544E4301) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], F = hd[2], C = hd[3];
    if (d != 64 || F <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights header\n"); return 3; }
    // Expected weight count: conv W (F*1*3*3) + conv b (F) + lin W (F*C) + lin b (C)
    long expected = 16 + 4L * (F * 9 + F + (long)F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_conv = p; p += F * 9;
    const float *b_conv = p; p += F;
    const float *W_lin = p; p += (long)F * C;
    const float *b_lin = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Temporary buffers
    float *conv_out = malloc(sizeof(float) * F * 8 * 8); // after conv+ReLU
    float *gap = malloc(sizeof(float) * F);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize input to float [0,1] (divide by 16)
        // We'll compute convolution directly with loops
        // Input is 8x8, single channel
        for (int f = 0; f < F; f++) {
            for (int oh = 0; oh < 8; oh++) {
                for (int ow = 0; ow < 8; ow++) {
                    float sum = b_conv[f];
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1; // padding=1
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8) {
                                float val = xi[ih * 8 + iw] / 16.0f;
                                sum += val * W_conv[(f * 9) + kh * 3 + kw];
                            }
                        }
                    }
                    // ReLU
                    conv_out[f * 64 + oh * 8 + ow] = sum > 0 ? sum : 0;
                }
            }
        }

        // Global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0;
            for (int idx = 0; idx < 64; idx++) sum += conv_out[f * 64 + idx];
            gap[f] = sum / 64.0f;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = b_lin[c];
            for (int f = 0; f < F; f++) sum += gap[f] * W_lin[f * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(conv_out); free(gap); free(logits);
    return 0;
}
```