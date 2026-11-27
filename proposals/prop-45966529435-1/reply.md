```hypothesis
Family: label-cnn. A tiny CNN with one 3x3 convolutional layer (8 filters), global average pooling, and a linear classifier. It exploits spatial locality in 8x8 digit images, which linear models ignore. With ~170 parameters it trains quickly and should infer at very low cost (similar to label-linear). If dev accuracy does not exceed label-linear (0.9613), the hypothesis fails.
```

```python
"""label-cnn: tiny CNN for 8x8 ordinal pixel digits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: 1 input channel -> 8 3x3 conv (pad=1) -> ReLU -> global avg pool -> 10-class linear.
Exported weights: magic 0x434E4E31, d=64, C=10, F=8, K=3; float32 mean[64], std[64], conv_w[8*3*3], conv_b[8], fc_w[8*10], fc_b[10]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
F, K = 8, 3          # filters, kernel size
H, W = 8, 8          # input height, width (8x8)
C = 10               # classes

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col(x, k, pad):
    """x: (N, 1, H, W) -> (N*H*W, k*k)"""
    N, _, H, W = x.shape
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, H, W, k*k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            cols[:, :, :, i*k+j] = x_pad[:, 0, i:i+H, j:j+W]
    return cols.reshape(-1, k*k)

def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64 and card["classes"] == 10
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Per-pixel z-scoring (ordinal)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_norm = (X.astype(np.float32) - mu) / sd
    X_img = X_norm.reshape(n, 1, H, W)   # (N,1,8,8)

    rng = np.random.default_rng(SEED)
    # He init for conv, Xavier for fc
    conv_w = (rng.normal(0, math.sqrt(2.0 / (K*K)), (F, K*K))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    fc_w = (rng.normal(0, math.sqrt(2.0 / F), (F, C))).astype(np.float32)
    fc_b = np.zeros(C, np.float32)
    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X_img[idx]          # (B,1,8,8)
            yy = y[idx]             # (B,)
            B = x.shape[0]

            # Forward
            col = im2col(x, K, 1)           # (B*64, 9)
            conv_out = col @ conv_w.T + conv_b   # (B*64, F)
            conv_out = conv_out.reshape(B, H, W, F).transpose(0,3,1,2)  # (B,F,8,8)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.mean(axis=(2,3))   # (B,F)
            logits = pool_out @ fc_w + fc_b       # (B,C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            probs /= B

            # Backward
            grad_fc_w = pool_out.T @ probs + L2 * fc_w   # (F,C)
            grad_fc_b = probs.sum(axis=0)                # (C,)
            grad_pool = probs @ fc_w.T                   # (B,F)
            grad_relu = grad_pool[:, :, None, None] * (conv_out > 0)  # (B,F,8,8)
            grad_relu = grad_relu.transpose(0,2,3,1).reshape(B*H*W, F)  # (B*64, F)
            grad_conv_w = grad_relu.T @ col + L2 * conv_w   # (F,9)
            grad_conv_b = grad_relu.sum(axis=0)             # (F,)

            grads = [grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x434E4E31  # 'CNN1'
    out = struct.pack("<5i", magic, 64, C, F, K)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += fc_w.astype("<f4").tobytes()
    out += fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 ordinal pixels). Output: 1 byte class index. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4];
    if (d != 64 || C != 10 || F != 8 || K != 3) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + (long)d*4*2 + (long)F*K*K*4 + (long)F*4 + (long)F*C*4 + (long)C*4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 5);
    const float *sd = mu + d;
    const float *conv_w = sd + d;
    const float *conv_b = conv_w + (long)F*K*K;
    const float *fc_w = conv_b + F;
    const float *fc_b = fc_w + (long)F*C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float col[64*9];          // im2col buffer for one sample (64 positions, 9 elements)
    float conv_out[8*8*8];    // 8 filters, 8x8
    float pool_out[8];        // after global avg pool
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and im2col (3x3 kernel, padding=1)
        for (int p = 0; p < 64; p++) {
            int r = p / 8, c = p % 8;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    int rr = r + kr - 1, cc = c + kc - 1;
                    float val = 0.0f;
                    if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
                        int idx = rr*8 + cc;
                        val = (xi[idx] - mu[idx]) / sd[idx];
                    }
                    col[p*9 + kr*3 + kc] = val;
                }
            }
        }

        // Convolution + ReLU
        for (int f = 0; f < F; f++) {
            const float *w = conv_w + f*9;
            for (int p = 0; p < 64; p++) {
                float sum = conv_b[f];
                for (int kk = 0; kk < 9; kk++) sum += col[p*9 + kk] * w[kk];
                conv_out[f*64 + p] = sum > 0 ? sum : 0;
            }
        }

        // Global average pool
        for (int f = 0; f < F; f++) {
            float sum = 0;
            for (int p = 0; p < 64; p++) sum += conv_out[f*64 + p];
            pool_out[f] = sum / 64.0f;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++) sum += pool_out[f] * fc_w[f*C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```