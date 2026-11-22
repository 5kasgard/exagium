```hypothesis
We propose a small CNN (label-cnn) that exploits spatial locality in the 8x8 digit images. A single 3x3 convolutional layer with 8 filters followed by global average pooling and a linear classifier should capture local stroke patterns better than a linear model, improving accuracy over label-linear while keeping inference cost low due to few parameters (~170 weights). If dev accuracy does not exceed label-linear (0.9613), the hypothesis fails.
```

```python
"""label-cnn: a tiny CNN for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: input 8x8x1 -> conv3x3 (8 filters, pad=1) -> ReLU -> global avg pool -> linear (10).
Per-pixel z-score normalization (mean/std from training)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
MOMENTUM = 0.9
L2 = 1e-4
BATCH = 64
EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col(x, kh, kw, pad=1):
    """x: (N, C, H, W) -> (N, OH*OW, C*kh*kw)"""
    N, C, H, W = x.shape
    OH = H + 2*pad - kh + 1
    OW = W + 2*pad - kw + 1
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, OH*OW, C*kh*kw), dtype=np.float32)
    for i in range(OH):
        for j in range(OW):
            patch = x_pad[:, :, i:i+kh, j:j+kw].reshape(N, C*kh*kw)
            cols[:, i*OW + j, :] = patch
    return cols, (OH, OW)

def col2im(cols, x_shape, kh, kw, pad=1):
    """cols: (N, OH*OW, C*kh*kw) -> (N, C, H, W)"""
    N, C, H, W = x_shape
    OH = H + 2*pad - kh + 1
    OW = W + 2*pad - kw + 1
    dx_pad = np.zeros((N, C, H+2*pad, W+2*pad), dtype=np.float32)
    for i in range(OH):
        for j in range(OW):
            patch = cols[:, i*OW + j, :].reshape(N, C, kh, kw)
            dx_pad[:, :, i:i+kh, j:j+kw] += patch
    return dx_pad[:, :, pad:pad+H, pad:pad+W]

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64

    rng = np.random.default_rng(SEED)

    # per-pixel normalization
    mean = X_bytes.mean(axis=0).astype(np.float32)
    std = X_bytes.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X = (X_bytes.astype(np.float32) - mean) / std
    X = X.reshape(-1, 1, 8, 8)  # (N, 1, 8, 8)

    # parameters
    F = 8  # conv filters
    KH = KW = 3
    # He init for conv
    conv_W = (rng.normal(0, math.sqrt(2.0 / (1*KH*KW)), size=(F, 1, KH, KW))).astype(np.float32)
    conv_b = np.zeros(F, dtype=np.float32)
    # Xavier init for fc
    fc_W = (rng.uniform(-math.sqrt(6.0/(F+C)), math.sqrt(6.0/(F+C)), size=(C, F))).astype(np.float32)
    fc_b = np.zeros(C, dtype=np.float32)

    # momentum buffers
    v_conv_W = np.zeros_like(conv_W)
    v_conv_b = np.zeros_like(conv_b)
    v_fc_W = np.zeros_like(fc_W)
    v_fc_b = np.zeros_like(fc_b)

    N = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            x = X[idx]          # (B,1,8,8)
            t = y[idx]          # (B,)
            B = x.shape[0]

            # forward
            # conv
            x_col, (OH, OW) = im2col(x, KH, KW, pad=1)  # (B, 64, 9)
            conv_out = x_col @ conv_W.reshape(F, -1).T + conv_b  # (B, 64, F)
            conv_out = conv_out.transpose(0, 2, 1).reshape(B, F, OH, OW)  # (B, F, 8, 8)
            # ReLU
            relu_out = np.maximum(conv_out, 0)
            # global avg pool
            pool_out = relu_out.mean(axis=(2,3))  # (B, F)
            # linear
            logits = pool_out @ fc_W.T + fc_b  # (B, C)

            # softmax + cross-entropy
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # gradient of loss w.r.t logits
            dlogits = probs
            dlogits[np.arange(B), t] -= 1.0
            dlogits /= B

            # backward
            # fc
            dfc_W = dlogits.T @ pool_out + L2 * fc_W
            dfc_b = dlogits.sum(axis=0)
            dpool = dlogits @ fc_W  # (B, F)

            # pool (global avg)
            drelu = np.repeat(np.repeat(dpool[:, :, None, None], OH, axis=2), OW, axis=3) / (OH*OW)

            # ReLU
            drelu[conv_out <= 0] = 0

            # conv
            dconv_out = drelu.transpose(0, 2, 3, 1).reshape(B, OH*OW, F)  # (B, 64, F)
            dconv_W = dconv_out.transpose(1, 2, 0) @ x_col  # (64, F, B) @ (B, 64, 9) -> (64, F, 9) -> sum over first dim?
            # Actually: dconv_W[f, c*kh*kw] = sum_{b,i} dconv_out[b, i, f] * x_col[b, i, c*kh*kw]
            # So dconv_W shape (F, 9)
            dconv_W = np.zeros((F, 9), dtype=np.float32)
            for f in range(F):
                dconv_W[f] = (dconv_out[:, :, f].T @ x_col).sum(axis=0)  # (9,)
            dconv_W += L2 * conv_W.reshape(F, -1)
            dconv_b = dconv_out.sum(axis=(0,1))

            # update with momentum
            for param, grad, vel in [
                (conv_W, dconv_W.reshape(conv_W.shape), v_conv_W),
                (conv_b, dconv_b, v_conv_b),
                (fc_W, dfc_W, v_fc_W),
                (fc_b, dfc_b, v_fc_b),
            ]:
                vel *= MOMENTUM
                vel += (1 - MOMENTUM) * grad
                param -= LR * vel

    # write weights.bin
    # magic 'CNN1' = 0x314E4E43 little-endian
    magic = 0x314E4E43
    header = struct.pack("<6i", magic, d, C, F, KH, KW)
    # mean[64], std[64], conv_W[F,1,3,3], conv_b[F], fc_W[C,F], fc_b[C]
    body = (mean.tobytes() + std.tobytes() +
            conv_W.astype("<f4").tobytes() + conv_b.astype("<f4").tobytes() +
            fc_W.astype("<f4").tobytes() + fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 pixels). Output: one byte class per record.
 * Weights format: magic 0x314E4E43, d, C, F, KH, KW;
 * float mean[d], std[d], conv_W[F][1][KH][KW], conv_b[F], fc_W[C][F], fc_b[C]. */
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
    uint8_t *wdata = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wdata;
    if (hdr[0] != 0x314E4E43) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], F = hdr[3], KH = hdr[4], KW = hdr[5];
    if (d != 64 || C != 10 || F != 8 || KH != 3 || KW != 3) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 24 + 4*d + 4*d + 4*F*KH*KW + 4*F + 4*C*F + 4*C;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    float *ptr = (float *)(wdata + 24);
    float *mean = ptr; ptr += d;
    float *std = ptr; ptr += d;
    float *conv_W = ptr; ptr += F*KH*KW;
    float *conv_b = ptr; ptr += F;
    float *fc_W = ptr; ptr += C*F;
    float *fc_b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_img = malloc(64 * sizeof(float));
    float *conv_out = malloc(8*8*8 * sizeof(float)); // F, OH, OW
    float *pool_out = malloc(8 * sizeof(float));
    float *logits = malloc(10 * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // normalize
        for (int p = 0; p < 64; p++) in_img[p] = (xi[p] - mean[p]) / std[p];

        // conv: 1 channel, 8 filters, 3x3, pad=1 -> 8x8x8
        // We'll compute directly: for each filter f, each output position (oh,ow)
        for (int f = 0; f < F; f++) {
            float bias = conv_b[f];
            const float *kw = conv_W + f*9;
            for (int oh = 0; oh < 8; oh++) {
                for (int ow = 0; ow < 8; ow++) {
                    float sum = bias;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh + kh - 1;
                        for (int kw_i = 0; kw_i < 3; kw_i++) {
                            int iw = ow + kw_i - 1;
                            if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8) {
                                float val = in_img[ih*8 + iw];
                                sum += val * kw[kh*3 + kw_i];
                            }
                        }
                    }
                    conv_out[(f*8 + oh)*8 + ow] = sum > 0 ? sum : 0; // ReLU fused
                }
            }
        }

        // global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0;
            for (int oh = 0; oh < 8; oh++)
                for (int ow = 0; ow < 8; ow++)
                    sum += conv_out[(f*8 + oh)*8 + ow];
            pool_out[f] = sum / 64.0f;
        }

        // linear
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++)
                sum += pool_out[f] * fc_W[c*F + f];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```