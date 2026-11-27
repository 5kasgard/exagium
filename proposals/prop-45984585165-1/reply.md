```hypothesis
The optdigits data are 8x8 images with strong local spatial correlations. A tiny CNN with 3x3 filters (8 filters) exploits this structure via weight sharing, giving higher accuracy than a linear model with far fewer parameters than a fully-connected hidden layer. The model should beat label-linear (0.9613) and approach label-mlp16 (0.9770) at lower inference cost because convolutions on 8x8 are extremely cheap. Failure would be dev accuracy ≤ 0.9613 or inference cost ≥ label-mlp16.
```
```python
"""CNN family for 8x8 optdigits: one 3x3 conv layer (H filters), ReLU, global avg pool, linear to 10 classes.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exported weights: magic 0x434E4E31, d=64, H, C=10; float32 mean[64], std[64], conv_w[H][3][3], conv_b[H], fc_w[H][C], fc_b[C]."""
import json, math, struct, sys

import numpy as np

H = 8               # number of 3x3 filters
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

def im2col(x, kh, kw):   # x: (N, 1, 8, 8) -> (N*36, 9)
    N = x.shape[0]
    col = np.zeros((N, 36, kh*kw), dtype=np.float32)
    idx = 0
    for i in range(6):
        for j in range(6):
            patch = x[:, 0, i:i+kh, j:j+kw].reshape(N, -1)
            col[:, idx, :] = patch
            idx += 1
    return col.reshape(-1, kh*kw)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 64

    rng = np.random.default_rng(SEED)

    # Normalize per pixel (mean, std over full training set)
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0) + 1e-6
    Xn = (Xf - mean) / std
    Xn = Xn.reshape(n, 1, 8, 8)   # (N, 1, 8, 8)

    # Initialize weights
    # conv: H filters, 1 input channel, 3x3
    fan_in_conv = 9
    scale_conv = math.sqrt(2.0 / fan_in_conv)
    conv_w = (rng.normal(0, scale_conv, (H, 1, 3, 3))).astype(np.float32)
    conv_b = np.zeros(H, dtype=np.float32)

    # fc: H -> C
    fan_in_fc = H
    fan_out_fc = C
    scale_fc = math.sqrt(6.0 / (fan_in_fc + fan_out_fc))
    fc_w = (rng.uniform(-scale_fc, scale_fc, (H, C))).astype(np.float32)
    fc_b = np.zeros(C, dtype=np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = Xn[idx]          # (B, 1, 8, 8)
            yy = y[idx]          # (B,)
            B = x.shape[0]

            # Forward
            col = im2col(x, 3, 3)                 # (B*36, 9)
            conv_out = col @ conv_w.reshape(9, H) + conv_b   # (B*36, H)
            conv_out = conv_out.reshape(B, 36, H)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.mean(axis=1)      # (B, H)
            logits = pool_out @ fc_w + fc_b       # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            dlogits = probs / B                    # (B, C)

            # Gradients
            dfc_w = pool_out.T @ dlogits + L2 * fc_w
            dfc_b = dlogits.sum(axis=0)
            dpool = dlogits @ fc_w.T               # (B, H)

            drelu = dpool[:, None, :] * (conv_out > 0)   # (B, 36, H)
            drelu_flat = drelu.reshape(-1, H)            # (B*36, H)
            dconv_w = col.T @ drelu_flat + L2 * conv_w.reshape(9, H)
            dconv_w = dconv_w.reshape(H, 1, 3, 3)
            dconv_b = drelu_flat.sum(axis=0)

            grads = [dconv_w, dconv_b, dfc_w, dfc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<4i", magic, d, H, C)
    # mean[64], std[64], conv_w[H*9], conv_b[H], fc_w[H*C], fc_b[C]
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
/* Tiny CNN for optdigits: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || C != 10 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected floats: mean[64], std[64], conv_w[H*9], conv_b[H], fc_w[H*C], fc_b[C]
    long expected_floats = 64 + 64 + H*9 + H + H*C + C;
    if (nw != 16 + expected_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mean = p; p += 64;
    const float *std  = p; p += 64;
    const float *conv_w = p; p += H*9;
    const float *conv_b = p; p += H;
    const float *fc_w   = p; p += H*C;
    const float *fc_b   = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xn = malloc(64 * sizeof(float));
    float *col = malloc(36 * 9 * sizeof(float));
    float *conv_out = malloc(36 * H * sizeof(float));
    float *pool_out = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize
        for (int j = 0; j < 64; j++) xn[j] = (xi[j] - mean[j]) / std[j];

        // im2col: 36 patches of 9 elements each
        int idx = 0;
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                for (int dr = 0; dr < 3; dr++) {
                    for (int dc = 0; dc < 3; dc++) {
                        int src = (r + dr) * 8 + (c + dc);
                        col[idx++] = xn[src];
                    }
                }
            }
        }

        // Convolution: col (36x9) * conv_w (9xH) -> conv_out (36xH)
        for (int p_idx = 0; p_idx < 36; p_idx++) {
            for (int h = 0; h < H; h++) {
                float sum = conv_b[h];
                for (int w = 0; w < 9; w++) {
                    sum += col[p_idx*9 + w] * conv_w[h*9 + w];
                }
                conv_out[p_idx*H + h] = sum > 0 ? sum : 0; // ReLU
            }
        }

        // Global average pooling over 36 positions
        for (int h = 0; h < H; h++) {
            float sum = 0;
            for (int p_idx = 0; p_idx < 36; p_idx++) sum += conv_out[p_idx*H + h];
            pool_out[h] = sum / 36.0f;
        }

        // Fully connected
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int h = 0; h < H; h++) sum += pool_out[h] * fc_w[h*C + c];
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