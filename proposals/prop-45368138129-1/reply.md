```hypothesis
Family: label-cnn (tiny CNN for 8x8 digits). Exploits 2D spatial structure of optdigits with a 3x3 conv (8 filters), ReLU, global average pooling, and 10-way linear head. Expects >0.97 dev accuracy (beating label-linear's 0.9613) at ~5000 Ir/item (similar to linear). Fails if accuracy <=0.962 or cost >8000 Ir/item.
```
```python
"""label-cnn: tiny CNN for 8x8 digit images (optdigits).
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: Input 8x8x1 -> Conv3x3 (8 filters) -> ReLU -> GlobalAvgPool -> FC(10) -> Softmax.
Training: Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1.
Weights: magic 0x434E4E31, float32 arrays: conv_w[8,1,3,3], conv_b[8], fc_w[8,10], fc_b[10]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
KH, KW, NF = 3, 3, 8  # kernel and number of filters
OH, OW = 8 - KH + 1, 8 - KW + 1  # 6,6
FC_IN, FC_OUT = NF, 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def im2col(X, kh, kw):
    # X: (N, H, W) -> (N*OH*OW, kh*kw)
    N, H, W = X.shape
    oh, ow = H - kh + 1, W - kw + 1
    col = np.zeros((N, oh, ow, kh * kw), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            col[:, :, :, i * kw + j] = X[:, i:i + oh, j:j + ow]
    return col.reshape(-1, kh * kw), (N, oh, ow)


def col2im(dcol, N, H, W, kh, kw):
    # dcol: (N*OH*OW, kh*kw) -> (N, H, W)
    oh, ow = H - kh + 1, W - kw + 1
    dX = np.zeros((N, H, W), dtype=np.float32)
    dcol = dcol.reshape(N, oh, ow, kh * kw)
    for i in range(kh):
        for j in range(kw):
            dX[:, i:i + oh, j:j + ow] += dcol[:, :, :, i * kw + j]
    return dX


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert C == 10
    N = X.shape[0]

    # Normalize to [0,1] and reshape to (N,8,8)
    X = (X.astype(np.float32) / 16.0).reshape(N, 8, 8)

    rng = np.random.default_rng(SEED)
    # Xavier init
    conv_w = (rng.uniform(-1, 1, (NF, KH * KW)) * math.sqrt(6.0 / (KH * KW + NF))).astype(np.float32)
    conv_b = np.zeros(NF, np.float32)
    fc_w = (rng.uniform(-1, 1, (FC_IN, FC_OUT)) * math.sqrt(6.0 / (FC_IN + FC_OUT))).astype(np.float32)
    fc_b = np.zeros(FC_OUT, np.float32)
    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / N))
    for _ in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            xcol, spatial = im2col(xb, KH, KW)          # (bsz*36, 9)
            conv_out = xcol @ conv_w.T + conv_b           # (bsz*36, 8)
            relu_out = np.maximum(conv_out, 0)
            # Global average pool: reshape to (bsz, 36, 8) -> mean over 36 -> (bsz, 8)
            gap_out = relu_out.reshape(bsz, OH * OW, NF).mean(axis=1)
            logits = gap_out @ fc_w + fc_b                # (bsz, 10)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)  # (bsz, 10)

            # Backward FC
            grad_fc_w = gap_out.T @ grad_logits + L2 * fc_w
            grad_fc_b = grad_logits.sum(axis=0)
            grad_gap = grad_logits @ fc_w.T                # (bsz, 8)

            # Backward GAP
            grad_relu = np.repeat(grad_gap[:, None, :], OH * OW, axis=1).reshape(-1, NF) / (OH * OW)

            # Backward ReLU
            grad_conv = grad_relu * (conv_out > 0).astype(np.float32)

            # Backward Conv
            grad_conv_w = xcol.T @ grad_conv + L2 * conv_w.T  # (9, 8) -> transpose later
            grad_conv_b = grad_conv.sum(axis=0)
            grad_xcol = grad_conv @ conv_w                   # (bsz*36, 9)
            grad_x = col2im(grad_xcol, bsz, 8, 8, KH, KW)    # not used further

            grads = [grad_conv_w.T, grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, NF, C, KH, KW, OH, OW
    header = struct.pack("<8i", 0x434E4E31, 64, NF, C, KH, KW, OH, OW)
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
 * Input record: 64 bytes (8x8 pixels 0..16). Output: one byte class 0..9.
 * Weights format: magic 0x434E4E31, int32 d,NF,C,KH,KW,OH,OW; float32 conv_w[NF][KH*KW], conv_b[NF], fc_w[NF][C], fc_b[C]. */
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
    if (nw < 32 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], NF = hd[2], C = hd[3], KH = hd[4], KW = hd[5], OH = hd[6], OW = hd[7];
    if (d != 64 || NF != 8 || C != 10 || KH != 3 || KW != 3 || OH != 6 || OW != 6) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 32 + 4L * (NF * KH * KW + NF + NF * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 8);
    const float *conv_w = p; p += NF * KH * KW;
    const float *conv_b = p; p += NF;
    const float *fc_w = p; p += NF * C;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(sizeof(float) * OH * OW * NF);
    float *gap_out = malloc(sizeof(float) * NF);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize input to [0,1] in float[8][8]
        float in[8][8];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                in[r][c] = xi[r * 8 + c] / 16.0f;

        // Conv + ReLU: for each filter, each output position
        for (int f = 0; f < NF; f++) {
            const float *w = conv_w + f * KH * KW;
            float b = conv_b[f];
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = b;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            sum += in[oh + kh][ow + kw] * w[kh * KW + kw];
                        }
                    }
                    conv_out[(oh * OW + ow) * NF + f] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // Global average pool over 6x6 = 36 positions
        for (int f = 0; f < NF; f++) {
            float sum = 0.0f;
            for (int p = 0; p < OH * OW; p++)
                sum += conv_out[p * NF + f];
            gap_out[f] = sum / (OH * OW);
        }

        // FC
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < NF; f++)
                sum += gap_out[f] * fc_w[f * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```