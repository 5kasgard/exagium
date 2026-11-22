```hypothesis
Family: label-cnn1 (tiny 1-layer CNN with 8 3x3 filters, global average pooling, and linear classification).
Why: The 8x8 digit images have local spatial structure; a small CNN captures translation-invariant features with few parameters, outperforming linear models while remaining extremely fast at inference (few hundred multiply-adds per item).
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear by >2x, the family does not meet its goal.
```

```python
"""Tiny CNN for 8x8 optdigits: 8 3x3 filters, ReLU, global average pooling, linear to 10 classes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, struct, sys, math

import numpy as np
from numpy.lib.stride_tricks import as_strided

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 200
MAGIC = 0x434E4E31  # "CNN1" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def extract_patches(images):  # images: (B, 1, 8, 8) -> (B, 64, 9)
    B = images.shape[0]
    padded = np.zeros((B, 1, 10, 10), dtype=np.float32)
    padded[:, :, 1:9, 1:9] = images
    s = padded.strides
    patches = as_strided(padded, shape=(B, 8, 8, 3, 3), strides=(s[0], s[2], s[3], s[2], s[3]))
    return patches.reshape(B, 64, 9)


def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 64
    C = card["classes"]         # 10
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Normalize to [0,1] and reshape to (N, 1, 8, 8)
    X = (X.astype(np.float32) / 16.0).reshape(n, 1, 8, 8)

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # conv: 8 filters, 1 channel, 3x3 -> (8, 9)
    fan_in = 9
    W_conv = (rng.uniform(-1, 1, (8, 9)) * math.sqrt(6.0 / fan_in)).astype(np.float32)
    b_conv = np.zeros(8, np.float32)
    # fc: 8 -> 10
    fan_in_fc = 8
    W_fc = (rng.uniform(-1, 1, (10, 8)) * math.sqrt(6.0 / fan_in_fc)).astype(np.float32)
    b_fc = np.zeros(10, np.float32)

    params = [W_conv, b_conv, W_fc, b_fc]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward
            patches = extract_patches(xb)                 # (B, 64, 9)
            conv_pre = patches @ W_conv.T + b_conv        # (B, 64, 8)
            conv_out = np.maximum(conv_pre, 0)            # ReLU
            pool = conv_out.mean(axis=1)                  # (B, 8)
            logits = pool @ W_fc.T + b_fc                 # (B, 10)

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)  # (B, 10)

            # Backward
            # fc
            g_W_fc = g_logits.T @ pool + L2 * W_fc        # (10, 8)
            g_b_fc = g_logits.sum(axis=0)                 # (10,)
            g_pool = g_logits @ W_fc                      # (B, 8)

            # pooling (average over 64 positions)
            g_conv_out = g_pool[:, None, :] / 64.0        # (B, 1, 8) -> broadcast to (B, 64, 8)
            g_conv_out = np.broadcast_to(g_conv_out, (B, 64, 8))

            # ReLU
            g_conv_pre = g_conv_out * (conv_pre > 0)

            # conv
            g_W_conv = np.zeros_like(W_conv)
            g_b_conv = np.zeros_like(b_conv)
            # For each filter, accumulate over batch and spatial positions
            # g_W_conv[f] = sum_{b,i} g_conv_pre[b,i,f] * patches[b,i]
            # We can do: (B*64, 9) and (B*64, 8)
            patches_flat = patches.reshape(-1, 9)         # (B*64, 9)
            g_conv_pre_flat = g_conv_pre.reshape(-1, 8)   # (B*64, 8)
            g_W_conv = g_conv_pre_flat.T @ patches_flat + L2 * W_conv  # (8, 9)
            g_b_conv = g_conv_pre_flat.sum(axis=0)        # (8,)

            grads = [g_W_conv, g_b_conv, g_W_fc, g_b_fc]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, num_filters, fc_out
    header = struct.pack("<3i", MAGIC, 8, 10)
    # Order: W_conv (8*9), b_conv (8), W_fc (10*8), b_fc (10)
    body = (W_conv.astype("<f4").tobytes() +
            b_conv.astype("<f4").tobytes() +
            W_fc.astype("<f4").tobytes() +
            b_fc.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Tiny CNN inference for 8x8 optdigits: 8 3x3 filters, ReLU, global avg pool, linear 8->10.
 * ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixels 0..16). Output: one byte class. */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int num_filters = hd[1];
    int fc_out = hd[2];
    if (num_filters != 8 || fc_out != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    int expected_floats = 8*9 + 8 + 10*8 + 10;  // 170
    if (nw != 12 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *weights = (float *)(wb + 12);
    float *W_conv = weights;                  // 8*9
    float *b_conv = W_conv + 8*9;             // 8
    float *W_fc = b_conv + 8;                 // 10*8
    float *b_fc = W_fc + 10*8;                // 10

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float pool[8];
    float logits[10];
    float conv_out[8][8][8];  // [filter][i][j] after ReLU

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize input to float[8][8]
        float in[8][8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                in[r][c] = xi[r*8 + c] / 16.0f;
            }
        }

        // Convolution with 3x3 filters, padding=1, stride=1 -> 8x8 per filter
        for (int f = 0; f < 8; f++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = b_conv[f];
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int rr = r + dr;
                            int cc = c + dc;
                            float val = 0.0f;
                            if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) val = in[rr][cc];
                            int w_idx = f*9 + (dr+1)*3 + (dc+1);
                            sum += val * W_conv[w_idx];
                        }
                    }
                    conv_out[f][r][c] = sum > 0.0f ? sum : 0.0f;  // ReLU
                }
            }
        }

        // Global average pooling over 8x8 for each filter
        for (int f = 0; f < 8; f++) {
            float s = 0.0f;
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) s += conv_out[f][r][c];
            pool[f] = s / 64.0f;
        }

        // Linear layer
        for (int c = 0; c < 10; c++) {
            float sum = b_fc[c];
            for (int f = 0; f < 8; f++) sum += pool[f] * W_fc[c*8 + f];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```