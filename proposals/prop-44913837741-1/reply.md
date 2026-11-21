```hypothesis
Family: label-cnn1 (tiny CNN: 8 3x3 filters, ReLU, global average pooling, linear).
Why: Exploits 2D spatial structure of 8x8 digits; similar compute to linear but nonlinear.
Failure: If dev accuracy <= 0.9613 or Ir/item >= 8000, it doesn't beat label-linear.
```
```python
import json, math, struct, sys
import numpy as np

F = 8
K = 3
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

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64
    n = X_bytes.shape[0]

    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    X_img = X_norm.reshape(n, 1, 8, 8)
    pad = 1
    X_pad = np.pad(X_img, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    col = np.zeros((n, 64, K*K), dtype=np.float32)
    for kh in range(K):
        for kw in range(K):
            slice_ = X_pad[:, :, kh:kh+8, kw:kw+8].reshape(n, 64)
            col[:, :, kh*K + kw] = slice_.astype(np.float32)

    rng = np.random.default_rng(SEED)
    W_conv = (rng.normal(0, math.sqrt(2.0 / (K*K)), (F, K*K))).astype(np.float32)
    b_conv = np.zeros(F, np.float32)
    W_fc = (rng.normal(0, math.sqrt(2.0 / (F + C)), (C, F))).astype(np.float32)
    b_fc = np.zeros(C, np.float32)

    params = [W_conv, b_conv, W_fc, b_fc]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    y_onehot = np.zeros((n, C), np.float32)
    y_onehot[np.arange(n), y] = 1.0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            col_b = col[idx]
            y_b = y_onehot[idx]
            bsz = len(idx)

            out_conv = col_b @ W_conv.T + b_conv
            out_relu = np.maximum(out_conv, 0)
            out_pool = out_relu.mean(axis=1)
            scores = out_pool @ W_fc.T + b_fc

            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            d_scores = (probs - y_b) / bsz

            d_W_fc = d_scores.T @ out_pool + L2 * W_fc
            d_b_fc = d_scores.sum(axis=0)
            d_pool = d_scores @ W_fc

            d_relu = d_pool[:, None, :] / 64.0
            d_relu = d_relu * (out_conv > 0)

            d_W_conv = np.zeros_like(W_conv)
            d_b_conv = np.zeros_like(b_conv)
            d_col = d_relu @ W_conv
            for f in range(F):
                d_W_conv[f] = col_b[:, :, :].transpose(1, 0) @ d_relu[:, :, f] + L2 * W_conv[f]
                d_b_conv[f] = d_relu[:, :, f].sum()

            grads = [d_W_conv, d_b_conv, d_W_fc, d_b_fc]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x434E4E31
    mean_f32 = mean.astype(np.float32)
    std_f32 = std.astype(np.float32)
    W_conv_f32 = W_conv.astype(np.float32)
    b_conv_f32 = b_conv.astype(np.float32)
    W_fc_f32 = W_fc.astype(np.float32)
    b_fc_f32 = b_fc.astype(np.float32)

    header = struct.pack("<5i", magic, d, C, F, K)
    body = mean_f32.tobytes() + std_f32.tobytes() + W_conv_f32.tobytes() + b_conv_f32.tobytes() + W_fc_f32.tobytes() + b_fc_f32.tobytes()
    open(sys.argv[4], "wb").write(header + body)

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4];
    if (d != 64 || C != 10 || F != 8 || K != 3) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20 + 4L * (64*2 + F*K*K + F + C*F + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wb + 20);
    float *mean = ptr; ptr += 64;
    float *std = ptr; ptr += 64;
    float *W_conv = ptr; ptr += F * K * K;
    float *b_conv = ptr; ptr += F;
    float *W_fc = ptr; ptr += C * F;
    float *b_fc = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float in_norm[64];
    float padded[10][10];
    float conv_out[8][8][8];
    float pool_out[8];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            in_norm[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int y = 0; y < 10; y++) for (int x_ = 0; x_ < 10; x_++) padded[y][x_] = 0.0f;
        for (int y = 0; y < 8; y++) for (int x_ = 0; x_ < 8; x_++) padded[y+1][x_+1] = in_norm[y*8 + x_];
        for (int f = 0; f < F; f++) {
            for (int y = 0; y < 8; y++) {
                for (int x_ = 0; x_ < 8; x_++) {
                    float sum = b_conv[f];
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            float w = W_conv[f * K * K + ky * K + kx];
                            sum += padded[y + ky][x_ + kx] * w;
                        }
                    }
                    conv_out[f][y][x_] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int y = 0; y < 8; y++) for (int x_ = 0; x_ < 8; x_++) sum += conv_out[f][y][x_];
            pool_out[f] = sum / 64.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b_fc[c];
            for (int f = 0; f < F; f++) {
                sum += pool_out[f] * W_fc[c * F + f];
            }
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```