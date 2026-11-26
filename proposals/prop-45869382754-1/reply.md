```hypothesis
Family: label-conv (single 3x3 conv layer + global average pooling + linear). Exploits 8x8 spatial structure of optdigits with weight sharing, giving stronger inductive bias than linear/MLP. Expects >0.97 dev accuracy at ~10k Ir/item (between label-linear and label-mlp16). Failure if accuracy ≤ label-linear (0.961) or cost ≥ label-mlp16 (8.7k Ir/item).
```
```python
"""label-conv: 3x3 conv + global avg pool + linear for 8x8 ordinal pixels.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Exports weights as:
int32 magic=0x434F4E56, d, H, C, k, stride, pad; float32 mean[d], std[d],
conv_w[H][k][k], conv_b[H], fc_w[H][C], fc_b[C]."""
import json, math, struct, sys

import numpy as np

H = 16          # number of filters
K = 3           # kernel size
STRIDE = 1
PAD = 0
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def im2col(x, k, stride, pad):
    """x: (N, H, W) -> (N, OH*OW, k*k)"""
    N, H, W = x.shape
    OH = (H + 2 * pad - k) // stride + 1
    OW = (W + 2 * pad - k) // stride + 1
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    cols = []
    for oh in range(OH):
        for ow in range(OW):
            h_start = oh * stride
            w_start = ow * stride
            patch = x[:, h_start:h_start + k, w_start:w_start + k]
            cols.append(patch.reshape(N, -1))
    return np.stack(cols, axis=1)  # (N, OH*OW, k*k)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    assert d == 64
    rng = np.random.default_rng(SEED)

    # per-pixel normalization (ordinal)
    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_bytes.astype(np.float32) - mu) / sd).reshape(-1, 8, 8)  # (N,8,8)

    n = X.shape[0]
    OH = (8 + 2 * PAD - K) // STRIDE + 1  # 6
    OW = OH
    P = OH * OW  # 36

    # init weights
    fan_in = K * K
    conv_w = (rng.normal(0, math.sqrt(2.0 / fan_in), (H, K, K))).astype(np.float32)
    conv_b = np.zeros(H, np.float32)
    fc_w = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]          # (B,8,8)
            yy = y[idx]         # (B,)
            B = x.shape[0]

            # forward
            cols = im2col(x, K, STRIDE, PAD)          # (B, P, K*K)
            conv_out = cols @ conv_w.reshape(H, -1).T + conv_b   # (B, P, H)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.mean(axis=1)          # (B, H)
            logits = pool_out @ fc_w + fc_b           # (B, C)

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # backward
            grad_fc_w = pool_out.T @ grad_logits + L2 * fc_w
            grad_fc_b = grad_logits.sum(axis=0)
            grad_pool = grad_logits @ fc_w.T               # (B, H)

            grad_relu = grad_pool[:, None, :] * (conv_out > 0)  # (B, P, H)
            grad_conv_b = grad_relu.sum(axis=(0, 1))
            grad_conv_w = np.zeros_like(conv_w)
            # accumulate over positions
            for p in range(P):
                col = cols[:, p, :]   # (B, K*K)
                g = grad_relu[:, p, :]  # (B, H)
                grad_conv_w += col.T @ g   # (K*K, H)
            grad_conv_w = grad_conv_w.T.reshape(H, K, K) + L2 * conv_w

            grads = [grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x434F4E56  # 'CONV'
    header = struct.pack("<7i", magic, d, H, C, K, STRIDE, PAD)
    floats = np.concatenate([
        mu, sd,
        conv_w.ravel(), conv_b,
        fc_w.ravel(), fc_b
    ]).astype("<f4")
    open(sys.argv[4], "wb").write(header + floats.tobytes())


if __name__ == "__main__":
    main()
```
```c
/* label-conv inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 ordinal pixels). Output: 1-byte class.
 * Weights format: int32 magic=0x434F4E56, d, H, C, k, stride, pad;
 * float32 mean[d], std[d], conv_w[H][k][k], conv_b[H], fc_w[H][C], fc_b[C]. */
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
    if (nw < 28 || hd[0] != 0x434F4E56) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], K = hd[4], stride = hd[5], pad = hd[6];
    if (d != 64 || C < 1 || C > 256 || K != 3 || stride != 1 || pad != 0) {
        fprintf(stderr, "bad weights config\n"); return 3;
    }
    int OH = (8 + 2 * pad - K) / stride + 1;  // 6
    int OW = OH;
    int P = OH * OW;  // 36
    long expected_floats = 2L * d + (long)H * K * K + H + (long)H * C + C;
    if (nw != 28 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *f = (const float *)(hd + 7);
    const float *mean = f; f += d;
    const float *std  = f; f += d;
    const float *conv_w = f; f += (long)H * K * K;
    const float *conv_b = f; f += H;
    const float *fc_w   = f; f += (long)H * C;
    const float *fc_b   = f;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_img = malloc(64 * sizeof(float));
    float *col = malloc(P * K * K * sizeof(float));
    float *conv_out = malloc(P * H * sizeof(float));
    float *pool_out = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize
        for (int j = 0; j < d; j++) in_img[j] = (xi[j] - mean[j]) / std[j];

        // im2col
        int idx = 0;
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int h_start = oh * stride;
                int w_start = ow * stride;
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int h = h_start + kh - pad;
                        int w = w_start + kw - pad;
                        float val = 0.0f;
                        if (h >= 0 && h < 8 && w >= 0 && w < 8) val = in_img[h * 8 + w];
                        col[idx++] = val;
                    }
                }
            }
        }

        // conv + bias
        for (int p = 0; p < P; p++) {
            for (int h = 0; h < H; h++) {
                float sum = conv_b[h];
                for (int kk = 0; kk < K * K; kk++) {
                    sum += col[p * K * K + kk] * conv_w[h * K * K + kk];
                }
                conv_out[p * H + h] = sum > 0 ? sum : 0;  // ReLU
            }
        }

        // global average pooling
        for (int h = 0; h < H; h++) {
            float sum = 0;
            for (int p = 0; p < P; p++) sum += conv_out[p * H + h];
            pool_out[h] = sum / P;
        }

        // linear
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int h = 0; h < H; h++) sum += pool_out[h] * fc_w[h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```