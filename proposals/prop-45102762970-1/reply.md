```hypothesis
Family: label-tinycnn — a tiny CNN with 8 3×3 filters, global average pooling, and a linear head.
Why: Optdigits are 8×8 images; local convolutional filters capture stroke patterns better than linear models while adding only ~300 parameters. Global pooling avoids dense layers, keeping inference fast.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or instructions/item ≥ 8685 (label-mlp16), the hypothesis is wrong.
```

```python
"""label-tinycnn: tiny CNN for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-pixel z-score → 8 3x3 conv filters (pad=1, stride=1) → ReLU → global avg pool → linear (10).
Exported weights: magic 0x434E4E54, int32 dims, then float32 arrays in order:
 mean[64], std[64], conv_w[8*3*3], conv_b[8], fc_w[8*10], fc_b[10]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
N_FILTERS, FILTER_SIZE, PADDING, STRIDE = 8, 3, 1, 1
IN_H, IN_W, IN_C = 8, 8, 1
OUT_H = (IN_H + 2*PADDING - FILTER_SIZE)//STRIDE + 1  # 8
OUT_W = (IN_W + 2*PADDING - FILTER_SIZE)//STRIDE + 1  # 8
POOL_OUT = N_FILTERS  # global average pooling
NUM_CLASSES = 10
MAGIC = 0x434E4E54  # 'TNNC' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out


def im2col(x, k, pad, stride):
    # x: (N, C, H, W) -> (N, C*k*k, OH*OW)
    N, C, H, W = x.shape
    OH = (H + 2*pad - k)//stride + 1
    OW = (W + 2*pad - k)//stride + 1
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, C, k, k, OH, OW), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:,:,i,j,:,:] = x_pad[:,:, i*stride:i*stride+OH*stride:stride, j*stride:j*stride+OW*stride:stride]
    return cols.reshape(N, C*k*k, OH*OW)


def col2im(cols, x_shape, k, pad, stride):
    # cols: (N, C*k*k, OH*OW) -> (N, C, H, W)
    N, C, H, W = x_shape
    OH = (H + 2*pad - k)//stride + 1
    OW = (W + 2*pad - k)//stride + 1
    x_pad = np.zeros((N, C, H+2*pad, W+2*pad), dtype=cols.dtype)
    cols_reshaped = cols.reshape(N, C, k, k, OH, OW)
    for i in range(k):
        for j in range(k):
            x_pad[:,:, i*stride:i*stride+OH*stride:stride, j*stride:j*stride+OW*stride:stride] += cols_reshaped[:,:,i,j,:,:]
    return x_pad[:,:,pad:pad+H, pad:pad+W]


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    assert d == 64
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # per-pixel z-score
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xn = (Xf - mean) / std
    Xn = Xn.reshape(n, 1, IN_H, IN_W)  # (N,1,8,8)

    rng = np.random.default_rng(SEED)
    # He init for conv (ReLU)
    fan_in = IN_C * FILTER_SIZE * FILTER_SIZE
    conv_w = (rng.normal(0, math.sqrt(2.0/fan_in), (N_FILTERS, IN_C, FILTER_SIZE, FILTER_SIZE))).astype(np.float32)
    conv_b = np.zeros(N_FILTERS, np.float32)
    # Xavier for fc
    fc_w = (rng.uniform(-1, 1, (POOL_OUT, NUM_CLASSES)) * math.sqrt(6.0/POOL_OUT)).astype(np.float32)
    fc_b = np.zeros(NUM_CLASSES, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = Xn[idx]          # (B,1,8,8)
            yy = y[idx]          # (B,)
            B = len(yy)

            # forward
            # conv
            cols = im2col(x, FILTER_SIZE, PADDING, STRIDE)          # (B, 1*3*3, 64)
            conv_w_flat = conv_w.reshape(N_FILTERS, -1)             # (8, 9)
            conv_out = cols @ conv_w_flat.T + conv_b                # (B, 64, 8) -> transpose to (B,8,64)
            conv_out = conv_out.transpose(0,2,1)                    # (B,8,64)
            relu_out = np.maximum(conv_out, 0)
            # global avg pool
            pool_out = relu_out.mean(axis=2)                        # (B,8)
            # linear
            logits = pool_out @ fc_w + fc_b                         # (B,10)

            # softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = probs / B                                 # (B,10)

            # backward linear
            grad_fc_w = pool_out.T @ grad_logits + L2 * fc_w        # (8,10)
            grad_fc_b = grad_logits.sum(axis=0)                     # (10,)
            grad_pool = grad_logits @ fc_w.T                        # (B,8)

            # backward pool (global avg)
            grad_relu = grad_pool[:,:,None] / (OUT_H*OUT_W)         # (B,8,1) -> broadcast to (B,8,64)
            grad_relu = np.broadcast_to(grad_relu, (B, N_FILTERS, OUT_H*OUT_W))

            # backward ReLU
            grad_conv = grad_relu * (conv_out > 0)                  # (B,8,64)

            # backward conv
            grad_conv = grad_conv.transpose(0,2,1)                  # (B,64,8)
            grad_conv_w_flat = cols.transpose(0,2,1) @ grad_conv    # (B,9,8) sum over batch
            grad_conv_w = grad_conv_w_flat.sum(axis=0).T.reshape(N_FILTERS, IN_C, FILTER_SIZE, FILTER_SIZE) + L2 * conv_w
            grad_conv_b = grad_conv.sum(axis=(0,1))                 # (8,)
            # no need for input gradient

            grads = [grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    # order: mean[64], std[64], conv_w[8*3*3], conv_b[8], fc_w[8*10], fc_b[10]
    header = struct.pack("<7i", MAGIC, N_FILTERS, FILTER_SIZE, IN_C, IN_H, IN_W, NUM_CLASSES)
    body = (mean.astype("<f4").tobytes() +
            std.astype("<f4").tobytes() +
            conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            fc_w.astype("<f4").tobytes() +
            fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-tinycnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8). Output: 1 byte class (0-9). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t*)wb;
    if (hd[0] != 0x434E4E54) { fprintf(stderr, "bad magic\n"); return 3; }
    int n_filters = hd[1], fsize = hd[2], in_c = hd[3], in_h = hd[4], in_w = hd[5], n_classes = hd[6];
    if (n_filters!=8 || fsize!=3 || in_c!=1 || in_h!=8 || in_w!=8 || n_classes!=10) {
        fprintf(stderr, "unexpected dims\n"); return 3;
    }
    int d = in_h * in_w;  // 64
    long expected = 28 + 4L*(2*d + n_filters*fsize*fsize*in_c + n_filters + n_filters*n_classes + n_classes);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *ptr = (const float*)(wb + 28);
    const float *mean = ptr; ptr += d;
    const float *std  = ptr; ptr += d;
    const float *conv_w = ptr; ptr += n_filters * fsize * fsize * in_c;
    const float *conv_b = ptr; ptr += n_filters;
    const float *fc_w   = ptr; ptr += n_filters * n_classes;
    const float *fc_b   = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(sizeof(float) * n_filters * in_h * in_w); // 8*8*8
    float *pool_out = malloc(sizeof(float) * n_filters);
    float *logits   = malloc(sizeof(float) * n_classes);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize and reshape to 8x8
        float in[64];
        for (int p = 0; p < d; p++) in[p] = (xi[p] - mean[p]) / std[p];

        // convolution: 8 filters, 3x3, pad=1, stride=1
        for (int f = 0; f < n_filters; f++) {
            const float *fw = conv_w + f * fsize * fsize;
            float b = conv_b[f];
            for (int oy = 0; oy < in_h; oy++) {
                for (int ox = 0; ox < in_w; ox++) {
                    float sum = b;
                    for (int ky = 0; ky < fsize; ky++) {
                        int iy = oy + ky - 1;
                        for (int kx = 0; kx < fsize; kx++) {
                            int ix = ox + kx - 1;
                            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                sum += in[iy * in_w + ix] * fw[ky * fsize + kx];
                            }
                        }
                    }
                    conv_out[f * in_h * in_w + oy * in_w + ox] = sum > 0 ? sum : 0; // ReLU
                }
            }
        }

        // global average pooling
        for (int f = 0; f < n_filters; f++) {
            float sum = 0;
            for (int p = 0; p < in_h * in_w; p++) sum += conv_out[f * in_h * in_w + p];
            pool_out[f] = sum / (in_h * in_w);
        }

        // linear
        for (int c = 0; c < n_classes; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < n_filters; f++) sum += pool_out[f] * fc_w[f * n_classes + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```