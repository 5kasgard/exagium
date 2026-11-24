```hypothesis
Family: label-tinycnn. A tiny CNN (conv3x3x8 -> ReLU -> 2x2 maxpool -> FC32 -> ReLU -> FC10) exploits the 8x8 spatial structure of optdigits. With ~4.5K params it should surpass linear (0.9613) and undercut mlp16 (8685 Ir/item) by using weight sharing and pooling. Fails if dev accuracy ≤ 0.9613 or Ir/item ≥ 8685.
```

```python
import json, math, struct, sys
import numpy as np
from numpy.lib.stride_tricks import as_strided

C1, K1 = 8, 3          # conv: 8 filters, 3x3
P2 = 2                 # maxpool 2x2
H3 = 32                # fc1 hidden
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def he_init(fan_in, fan_out, rng):
    return (rng.normal(0, math.sqrt(2.0 / fan_in), (fan_in, fan_out))).astype(np.float32)

def im2col(x, kh, kw, pad=1, stride=1):
    N, C, H, W = x.shape
    H_out = (H + 2*pad - kh) // stride + 1
    W_out = (W + 2*pad - kw) // stride + 1
    if pad > 0:
        x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    s = x.strides
    cols = as_strided(x, shape=(N, C, H_out, W_out, kh, kw),
                      strides=(s[0], s[1], s[2]*stride, s[3]*stride, s[2], s[3]))
    return cols.transpose(0, 2, 3, 1, 4, 5).reshape(N, H_out, W_out, -1)

def maxpool2d(x, k=2):
    N, C, H, W = x.shape
    return x.reshape(N, C, H//k, k, W//k, k).max(axis=(3, 5))

def maxpool2d_backward(grad_out, x, k=2):
    N, C, H, W = x.shape
    grad_in = np.zeros_like(x)
    x_rs = x.reshape(N, C, H//k, k, W//k, k)
    max_vals = x_rs.max(axis=(3, 5), keepdims=True)
    mask = (x_rs == max_vals)
    count = mask.sum(axis=(3, 5), keepdims=True)
    grad_in_rs = grad_in.reshape(N, C, H//k, k, W//k, k)
    grad_in_rs += mask * grad_out[:, :, :, None, :, None] / count
    return grad_in

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    N, D = X_bytes.shape
    assert D == 64

    X = (X_bytes.astype(np.float32) - 8.0) / 8.0
    X = X.reshape(N, 1, 8, 8)

    rng = np.random.default_rng(SEED)

    fan_in = 1*3*3
    conv1_w = rng.normal(0, math.sqrt(2.0 / fan_in), (C1, 1, K1, K1)).astype(np.float32)
    conv1_b = np.zeros(C1, np.float32)
    conv1_w_flat = conv1_w.reshape(C1, -1).T

    fc1_w = he_init(128, H3, rng)
    fc1_b = np.zeros(H3, np.float32)
    fc2_w = he_init(H3, C, rng)
    fc2_b = np.zeros(C, np.float32)

    params = [conv1_w, conv1_b, fc1_w, fc1_b, fc2_w, fc2_b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def softmax_crossentropy(logits, y):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        grad = probs.copy()
        grad[np.arange(len(y)), y] -= 1.0
        grad /= len(y)
        return loss, grad.astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / N))
    for epoch in range(max_epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            cols = im2col(xb, 3, 3, pad=1, stride=1)
            conv_out = (cols @ conv1_w_flat + conv1_b).transpose(0, 3, 1, 2)
            relu1 = np.maximum(conv_out, 0)
            pool_out = maxpool2d(relu1, 2)
            flat = pool_out.reshape(B, -1)
            fc1_out = flat @ fc1_w + fc1_b
            relu2 = np.maximum(fc1_out, 0)
            logits = relu2 @ fc2_w + fc2_b

            loss, grad_logits = softmax_crossentropy(logits, yb)

            grad_fc2_w = relu2.T @ grad_logits + L2 * fc2_w
            grad_fc2_b = grad_logits.sum(0)
            grad_relu2 = grad_logits @ fc2_w.T

            grad_fc1_out = grad_relu2 * (fc1_out > 0)
            grad_fc1_w = flat.T @ grad_fc1_out + L2 * fc1_w
            grad_fc1_b = grad_fc1_out.sum(0)
            grad_flat = grad_fc1_out @ fc1_w.T

            grad_pool = grad_flat.reshape(B, 8, 4, 4)
            grad_relu1 = maxpool2d_backward(grad_pool, relu1, 2)
            grad_conv = grad_relu1 * (conv_out > 0)

            grad_conv_t = grad_conv.transpose(0, 2, 3, 1)
            grad_conv1_w_flat = np.zeros((9, C1), np.float32)
            for b in range(B):
                grad_conv1_w_flat += cols[b].T @ grad_conv_t[b]
            grad_conv1_w_flat += L2 * conv1_w_flat
            grad_conv1_b = grad_conv.sum(axis=(0, 2, 3))
            grad_conv1_w = grad_conv1_w_flat.T.reshape(C1, 1, 3, 3)

            grads = [grad_conv1_w, grad_conv1_b, grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    header = struct.pack("<6i", 0x314E4E43, 64, C1, K1, H3, C)
    w_bytes = header
    w_bytes += conv1_w.astype("<f4").tobytes()
    w_bytes += conv1_b.astype("<f4").tobytes()
    w_bytes += fc1_w.astype("<f4").tobytes()
    w_bytes += fc1_b.astype("<f4").tobytes()
    w_bytes += fc2_w.astype("<f4").tobytes()
    w_bytes += fc2_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(w_bytes)

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

static float maxf(float a, float b) { return a > b ? a : b; }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint8_t *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x314E4E43) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C1 = hd[2], K1 = hd[3], H3 = hd[4], C = hd[5];
    if (d != 64 || C1 != 8 || K1 != 3 || H3 != 32 || C != 10) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 24 + (8*9 + 8 + 128*32 + 32 + 32*10 + 10) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wb + 24);
    float *conv1_w = ptr; ptr += 8*9;
    float *conv1_b = ptr; ptr += 8;
    float *fc1_w = ptr; ptr += 128*32;
    float *fc1_b = ptr; ptr += 32;
    float *fc2_w = ptr; ptr += 32*10;
    float *fc2_b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float conv_out[8][8][8];
    float relu1[8][8][8];
    float pool_out[8][4][4];
    float flat[128];
    float fc1_out[32];
    float relu2[32];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        float input[8][8];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                input[r][c] = (xi[r*8 + c] - 8.0f) / 8.0f;

        for (int oc = 0; oc < 8; oc++) {
            for (int oh = 0; oh < 8; oh++) {
                for (int ow = 0; ow < 8; ow++) {
                    float sum = conv1_b[oc];
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            float val = 0.0f;
                            if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8)
                                val = input[ih][iw];
                            sum += val * conv1_w[oc*9 + kh*3 + kw];
                        }
                    }
                    conv_out[oc][oh][ow] = sum;
                    relu1[oc][oh][ow] = sum > 0 ? sum : 0;
                }
            }
        }

        for (int c = 0; c < 8; c++)
            for (int ph = 0; ph < 4; ph++)
                for (int pw = 0; pw < 4; pw++) {
                    float m = relu1[c][2*ph][2*pw];
                    m = maxf(m, relu1[c][2*ph][2*pw+1]);
                    m = maxf(m, relu1[c][2*ph+1][2*pw]);
                    m = maxf(m, relu1[c][2*ph+1][2*pw+1]);
                    pool_out[c][ph][pw] = m;
                }

        int idx = 0;
        for (int c = 0; c < 8; c++)
            for (int ph = 0; ph < 4; ph++)
                for (int pw =