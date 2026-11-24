```hypothesis
We propose a small convolutional neural network family "label-cnn". CNNs exploit spatial locality in 8x8 digit images, which should improve accuracy over linear models. The architecture: 8 3x3 filters (pad=1), ReLU, 2x2 max-pool, then a fully-connected layer to 10 classes. If dev accuracy does not exceed label-linear (0.9613), the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
NUM_FILTERS = 8
FILTER_SIZE = 3
POOL_SIZE = 2
PAD = 1
STRIDE = 1
IN_CH = 1
IN_H, IN_W = 8, 8
OUT_H = IN_H + 2*PAD - FILTER_SIZE + 1  # 8
OUT_W = IN_W + 2*PAD - FILTER_SIZE + 1  # 8
POOL_H = OUT_H // POOL_SIZE  # 4
POOL_W = OUT_W // POOL_SIZE  # 4
FC_IN = NUM_FILTERS * POOL_H * POOL_W  # 128
CLASSES = 10

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col(x_pad, kh, kw, oh, ow):
    # x_pad: (B, C, H+2*pad, W+2*pad)
    B, C, _, _ = x_pad.shape
    col = np.zeros((B, oh, ow, C, kh, kw), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            col[:, i, j] = x_pad[:, :, i:i+kh, j:j+kw]
    return col.reshape(B*oh*ow, C*kh*kw)

def col2im(dcol, B, C, H, W, kh, kw, oh, ow, pad):
    # dcol: (B*oh*ow, C*kh*kw)
    dcol = dcol.reshape(B, oh, ow, C, kh, kw)
    dx_pad = np.zeros((B, C, H+2*pad, W+2*pad), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            dx_pad[:, :, i:i+oh, j:j+ow] += dcol[:, :, :, :, i, j]
    return dx_pad[:, :, pad:pad+H, pad:pad+W]

def maxpool_forward(x, pool_size):
    # x: (B, C, H, W) where H,W divisible by pool_size
    B, C, H, W = x.shape
    ph = pw = pool_size
    oh, ow = H // ph, W // pw
    x_reshaped = x.reshape(B, C, oh, ph, ow, pw)
    out = x_reshaped.max(axis=(3,5))
    # argmax for backward
    argmax = x_reshaped.argmax(axis=(3,5))  # not used directly, we need indices
    # compute flat indices within each pool window
    max_idx = x_reshaped.reshape(B, C, oh, ow, -1).argmax(axis=-1)
    return out, max_idx

def maxpool_backward(dout, max_idx, x_shape, pool_size):
    B, C, H, W = x_shape
    ph = pw = pool_size
    oh, ow = H // ph, W // pw
    dx = np.zeros(x_shape, dtype=np.float32)
    dout_flat = dout.reshape(B, C, oh, ow, 1)
    # scatter gradients to max positions
    for b in range(B):
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    idx = max_idx[b, c, i, j]
                    hi = idx // pw
                    wi = idx % pw
                    dx[b, c, i*ph+hi, j*pw+wi] += dout[b, c, i, j]
    return dx

def relu_forward(x):
    return np.maximum(x, 0)

def relu_backward(dout, x):
    return dout * (x > 0)

def linear_forward(x, W, b):
    return x @ W + b

def linear_backward(dout, x, W):
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db

def softmax_cross_entropy(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1
    dlogits /= B
    return loss, dlogits

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    assert d == 64
    C = card["classes"]
    assert C == 10

    # Normalize to [0,1]
    X = X.astype(np.float32) / 16.0
    X = X.reshape(n, 1, 8, 8)

    rng = np.random.default_rng(SEED)

    # Initialize weights
    fan_in_conv = IN_CH * FILTER_SIZE * FILTER_SIZE
    conv_w = rng.normal(0, np.sqrt(2.0 / fan_in_conv), (NUM_FILTERS, IN_CH, FILTER_SIZE, FILTER_SIZE)).astype(np.float32)
    conv_b = np.zeros(NUM_FILTERS, dtype=np.float32)
    fan_in_fc = FC_IN
    fc_w = rng.normal(0, np.sqrt(2.0 / fan_in_fc), (FC_IN, C)).astype(np.float32)
    fc_b = np.zeros(C, dtype=np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = x_batch.shape[0]

            # Forward
            # Conv
            x_pad = np.pad(x_batch, ((0,0),(0,0),(PAD,PAD),(PAD,PAD)), mode='constant')
            col = im2col(x_pad, FILTER_SIZE, FILTER_SIZE, OUT_H, OUT_W)
            conv_out = col @ conv_w.reshape(-1, NUM_FILTERS).T + conv_b  # (B*OH*OW, NF)
            conv_out = conv_out.reshape(B, OUT_H, OUT_W, NUM_FILTERS).transpose(0,3,1,2)  # (B, NF, OH, OW)

            # ReLU
            relu_out = relu_forward(conv_out)

            # MaxPool
            pool_out, pool_idx = maxpool_forward(relu_out, POOL_SIZE)

            # Flatten
            flat = pool_out.reshape(B, -1)

            # FC
            logits = linear_forward(flat, fc_w, fc_b)

            # Loss
            loss, dlogits = softmax_cross_entropy(logits, y_batch)

            # Backward
            dflat, dfc_w, dfc_b = linear_backward(dlogits, flat, fc_w)
            dpool = dflat.reshape(pool_out.shape)
            drelu = maxpool_backward(dpool, pool_idx, relu_out.shape, POOL_SIZE)
            dconv = relu_backward(drelu, conv_out)
            # dconv shape (B, NF, OH, OW) -> transpose to (B, OH, OW, NF) -> reshape (B*OH*OW, NF)
            dconv_flat = dconv.transpose(0,2,3,1).reshape(B*OUT_H*OUT_W, NUM_FILTERS)
            dconv_w = col.T @ dconv_flat  # (C*KH*KW, NF)
            dconv_b = dconv_flat.sum(axis=0)
            dcol = dconv_flat @ conv_w.reshape(-1, NUM_FILTERS)  # (B*OH*OW, C*KH*KW)
            dx_pad = col2im(dcol, B, IN_CH, IN_H, IN_W, FILTER_SIZE, FILTER_SIZE, OUT_H, OUT_W, PAD)
            # dx_pad is gradient w.r.t. padded input; we don't need dx for input

            # Add L2 regularization gradients
            dfc_w += L2 * fc_w
            dconv_w += L2 * conv_w.reshape(-1, NUM_FILTERS).T  # note shape: conv_w is (NF, C, KH, KW), we need (C*KH*KW, NF)
            # Actually dconv_w shape is (C*KH*KW, NF), conv_w reshaped is (NF, C*KH*KW).T -> (C*KH*KW, NF). Good.

            grads = [dconv_w.T.reshape(NUM_FILTERS, IN_CH, FILTER_SIZE, FILTER_SIZE), dconv_b, dfc_w, dfc_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Magic: 0x434E4E31 ('CNN1')
    # Header: magic, d, num_filters, filter_size, pool_size, fc_in, C
    header = struct.pack("<7i", 0x434E4E31, d, NUM_FILTERS, FILTER_SIZE, POOL_SIZE, FC_IN, C)
    # Weights order: conv_w (NF, IN_CH, KH, KW), conv_b (NF), fc_w (FC_IN, C), fc_b (C)
    # All float32 little-endian
    w_bytes = b""
    w_bytes += conv_w.astype("<f4").tobytes()
    w_bytes += conv_b.astype("<f4").tobytes()
    w_bytes += fc_w.astype("<f4").tobytes()
    w_bytes += fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + w_bytes)

if __name__ == "__main__":
    main()
```
```c
/* CNN family for optdigits: ./infer weights.bin in.rec N */
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

#define IN_CH 1
#define IN_H 8
#define IN_W 8
#define NUM_FILTERS 8
#define FILTER_SIZE 3
#define PAD 1
#define STRIDE 1
#define OUT_H (IN_H + 2*PAD - FILTER_SIZE + 1)  // 8
#define OUT_W (IN_W + 2*PAD - FILTER_SIZE + 1)  // 8
#define POOL_SIZE 2
#define POOL_H (OUT_H / POOL_SIZE)  // 4
#define POOL_W (OUT_W / POOL_SIZE)  // 4
#define FC_IN (NUM_FILTERS * POOL_H * POOL_W)  // 128
#define CLASSES 10

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 28 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], nf = hd[2], fs = hd[3], ps = hd[4], fc_in = hd[5], C = hd[6];
    if (d != 64 || nf != NUM_FILTERS || fs != FILTER_SIZE || ps != POOL_SIZE || fc_in != FC_IN || C != CLASSES) {
        fprintf(stderr, "weights architecture mismatch\n"); return 3;
    }
    long expected = 28 + 4L * (nf * IN_CH * fs * fs + nf + fc_in * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(hd + 7);
    const float *conv_w = ptr; ptr += nf * IN_CH * fs * fs;
    const float *conv_b = ptr; ptr += nf;
    const float *fc_w = ptr; ptr += fc_in * C;
    const float *fc_b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(sizeof(float) * NUM_FILTERS * OUT_H * OUT_W);
    float *relu_out = malloc(sizeof(float) * NUM_FILTERS * OUT_H * OUT_W);
    float *pool_out = malloc(sizeof(float) * NUM_FILTERS * POOL_H * POOL_W);
    float *fc_out = malloc(sizeof(float) * CLASSES);
    float *x_pad = malloc(sizeof(float) * IN_CH * (IN_H + 2*PAD) * (IN_W + 2*PAD));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and pad input
        memset(x_pad, 0, sizeof(float) * IN_CH * (IN_H + 2*PAD) * (IN_W + 2*PAD));
        for (int c = 0; c < IN_CH; c++) {
            for (int h = 0; h < IN_H; h++) {
                for (int w = 0; w < IN_W; w++) {
                    int idx = c * IN_H * IN_W + h * IN_W + w;
                    int pidx = c * (IN_H+2*PAD) * (IN_W+2*PAD) + (h+PAD) * (IN_W+2*PAD) + (w+PAD);
                    x_pad[pidx] = xi[idx] / 16.0f;
                }
            }
        }

        // Conv
        for (int f = 0; f < NUM_FILTERS; f++) {
            for (int oh = 0; oh < OUT_H; oh++) {
                for (int ow = 0; ow < OUT_W; ow++) {
                    float sum = conv_b[f];
                    for (int c = 0; c < IN_CH; c++) {
                        for (int kh = 0; kh < FILTER_SIZE; kh++) {
                            for (int kw = 0; kw < FILTER_SIZE; kw++) {
                                int in_h = oh + kh;
                                int in_w = ow + kw;
                                int in_idx = c * (IN_H+2*PAD) * (IN_W+2*PAD) + in_h * (IN_W+2*PAD) + in_w;
                                int w_idx = ((f * IN_CH + c) * FILTER_SIZE + kh) * FILTER_SIZE + kw;
                                sum += x_pad[in_idx] * conv_w[w_idx];
                            }
                        }
                    }
                    int out_idx = f * OUT_H * OUT_W + oh * OUT_W + ow;
                    conv_out[out_idx] = sum;
                }
            }
        }

        // ReLU
        for (int j = 0; j < NUM_FILTERS * OUT_H * OUT_W; j++) {
            relu_out[j] = conv_out[j] > 0 ? conv_out[j] : 0;
        }

        // MaxPool
        for (int f = 0; f < NUM_FILTERS; f++) {
            for (int ph = 0; ph < POOL_H; ph++) {
                for (int pw = 0; pw < POOL_W; pw++) {
                    float maxv = -1e30f;
                    for (int kh = 0; kh < POOL_SIZE; kh++) {
                        for (int kw = 0; kw < POOL_SIZE; kw++) {
                            int h = ph * POOL_SIZE + kh;
                            int w = pw * POOL_SIZE + kw;
                            int idx = f * OUT_H * OUT_W + h * OUT_W + w;
                            if (relu_out[idx] > maxv) maxv = relu_out[idx];
                        }
                    }
                    int pidx = f * POOL_H * POOL_W + ph * POOL_W + pw;
                    pool_out[pidx] = maxv;
                }
            }
        }

        // FC
        for (int c = 0; c < CLASSES; c++) {
            float sum = fc_b[c];
            for (int j = 0; j < FC_IN; j++) {
                sum += pool_out[j] * fc_w[j * CLASSES + c];
            }
            fc_out[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < CLASSES; c++) {
            if (fc_out[c] > fc_out[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```