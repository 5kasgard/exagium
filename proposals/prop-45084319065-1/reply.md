```hypothesis
Family: label-cnn (tiny 1-conv-layer CNN). The 8x8 optdigits images have strong local structure; a small CNN with 8 3x3 filters, ReLU, global average pooling, and a linear head should capture spatial patterns better than a linear model. With ~170 parameters it trains fast and infers with low CPU cost. If dev accuracy does not exceed label-linear (0.9613), it indicates the task gains little from local weight sharing or the model is under‑capacity.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 1e-3
BATCH = 64
EPOCHS = 50
NUM_FILTERS = 8
KERNEL_SIZE = 3
PADDING = 1
L2 = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = X.shape[0]

    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    X_img = X_norm.reshape(-1, 1, 8, 8)

    rng = np.random.default_rng(SEED)
    conv_w = rng.normal(0, math.sqrt(2.0 / (1 * 3 * 3)), (NUM_FILTERS, 1, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32)
    conv_b = np.zeros(NUM_FILTERS, dtype=np.float32)
    linear_w = rng.normal(0, math.sqrt(2.0 / NUM_FILTERS), (NUM_FILTERS, C)).astype(np.float32)
    linear_b = np.zeros(C, dtype=np.float32)

    params = [conv_w, conv_b, linear_w, linear_b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X_img[idx]
            yy = y[idx]
            b = x.shape[0]

            x_pad = np.pad(x, ((0,0),(0,0),(PADDING,PADDING),(PADDING,PADDING)), mode='constant')
            out_conv = np.zeros((b, NUM_FILTERS, 8, 8), dtype=np.float32)
            for f in range(NUM_FILTERS):
                for i in range(8):
                    for j in range(8):
                        window = x_pad[:, 0, i:i+3, j:j+3]
                        out_conv[:, f, i, j] = np.sum(window * conv_w[f, 0], axis=(1,2)) + conv_b[f]
            out_relu = np.maximum(out_conv, 0)
            gap = out_relu.mean(axis=(2,3))
            logits = gap @ linear_w + linear_b

            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            dlogits = probs.copy()
            dlogits[np.arange(b), yy] -= 1.0
            dlogits /= b

            dlinear_w = gap.T @ dlogits + L2 * linear_w
            dlinear_b = dlogits.sum(axis=0)
            dgap = dlogits @ linear_w.T

            drelu = dgap[:, :, None, None] * (out_conv > 0)

            dconv_w = np.zeros_like(conv_w)
            dconv_b = np.zeros_like(conv_b)
            for f in range(NUM_FILTERS):
                for i in range(8):
                    for j in range(8):
                        window = x_pad[:, 0, i:i+3, j:j+3]
                        grad = drelu[:, f, i, j][:, None, None]
                        dconv_w[f, 0] += np.sum(window * grad, axis=0)
                        dconv_b[f] += np.sum(grad)
            dconv_w += L2 * conv_w

            grads = [dconv_w, dconv_b, dlinear_w, dlinear_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    magic = 0x434E4E01
    out = struct.pack("<5i", magic, d, C, NUM_FILTERS, KERNEL_SIZE)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += linear_w.astype("<f4").tobytes()
    out += linear_b.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x434E4E01) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4];
    if (d != 64 || C != 10 || F != 8 || K != 3) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20 + 64*4 + 64*4 + 72*4 + 8*4 + 80*4 + 10*4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *mean = (float *)(hd + 5);
    float *std = mean + 64;
    float *conv_w = (float *)(std + 64);
    float *conv_b = conv_w + 72;
    float *linear_w = conv_b + 8;
    float *linear_b = linear_w + 80;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float input[64];
    float padded[10][10];
    float conv_out[8][8][8];
    float gap[8];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        for (int j = 0; j < 64; j++) {
            input[j] = (xi[j] - mean[j]) / std[j];
        }
        memset(padded, 0, sizeof(padded));
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                padded[r+1][c+1] = input[r*8 + c];
            }
        }
        for (int f = 0; f < 8; f++) {
            const float *w = conv_w + f * 9;
            float b = conv_b[f];
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = b;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            sum += padded[r+kr][c+kc] * w[kr*3 + kc];
                        }
                    }
                    conv_out[f][r][c] = sum;
                }
            }
        }
        for (int f = 0; f < 8; f++) {
            float sum = 0.0f;
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float v = conv_out[f][r][c];
                    if (v < 0) v = 0;
                    sum += v;
                }
            }
            gap[f] = sum / 64.0f;
        }
        for (int c = 0; c < 10; c++) {
            float sum = linear_b[c];
            for (int f = 0; f < 8; f++) {
                sum += gap[f] * linear_w[f*10 + c];
            }
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) {
            if (logits[c] > logits[best]) best = c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```