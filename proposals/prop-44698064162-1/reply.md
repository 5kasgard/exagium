```hypothesis
Family: label-tinycnn. A tiny CNN (8 filters, 3x3, padding 1, global average pooling, linear) exploits the 2D structure of 8x8 optdigits. With ~200 parameters it should beat label-linear (96.1%) while staying fast. If dev accuracy ≤ 96.1% or inference cost exceeds label-linear, the spatial inductive bias doesn't help here.
```

```python
"""Tiny CNN for 8x8 optdigits: 8 filters 3x3, ReLU, global average pool, linear to 10.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed architecture
FILTERS = 8
KERNEL = 3
PAD = 1
STRIDE = 1
IN_H, IN_W, IN_C = 8, 8, 1
OUT_H = (IN_H + 2*PAD - KERNEL) // STRIDE + 1  # 8
OUT_W = (IN_W + 2*PAD - KERNEL) // STRIDE + 1  # 8
POOL_SIZE = OUT_H * OUT_W  # 64
CLASSES = 10

# Training hyperparameters (tuned on tuning split)
SEED = 1
BATCH = 64
EPOCHS_SPLIT = 30
EPOCHS_FULL = 60
LR_CANDIDATES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

MAGIC = 0x434E4E31  # 'CNN1' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def normalize(X, mean, std):
    return (X - mean) / std


def im2col(X, kernel, pad, stride):
    # X: (N, H, W, C) -> (N, OH*OW, K*K*C)
    N, H, W, C = X.shape
    OH = (H + 2*pad - kernel) // stride + 1
    OW = (W + 2*pad - kernel) // stride + 1
    X_padded = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    col = np.zeros((N, OH*OW, kernel*kernel*C), dtype=np.float32)
    for i in range(OH):
        for j in range(OW):
            h_start = i * stride
            w_start = j * stride
            patch = X_padded[:, h_start:h_start+kernel, w_start:w_start+kernel, :]
            col[:, i*OW + j, :] = patch.reshape(N, -1)
    return col


def forward(X, W_conv, b_conv, W_lin, b_lin):
    # X: (N, 8, 8, 1)
    N = X.shape[0]
    col = im2col(X, KERNEL, PAD, STRIDE)          # (N, 64, 9)
    conv_out = col @ W_conv.T + b_conv             # (N, 64, F)
    relu_out = np.maximum(conv_out, 0)
    pool_out = relu_out.mean(axis=1)               # (N, F)
    logits = pool_out @ W_lin.T + b_lin            # (N, 10)
    return logits, pool_out, relu_out, col


def softmax_cross_entropy(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(N), y] -= 1.0
    grad /= N
    return loss, grad, probs


def backward(grad_logits, pool_out, relu_out, col, W_lin, W_conv, X, y, N, L2):
    # grad_logits: (N, 10)
    grad_W_lin = grad_logits.T @ pool_out + L2 * W_lin   # (10, F)
    grad_b_lin = grad_logits.sum(axis=0)                 # (10,)
    grad_pool = grad_logits @ W_lin                      # (N, F)
    grad_relu = grad_pool[:, None, :] / POOL_SIZE        # (N, 1, F) -> broadcast to (N, 64, F)
    grad_relu = np.repeat(grad_relu, POOL_SIZE, axis=1)  # (N, 64, F)
    grad_relu[relu_out <= 0] = 0
    grad_W_conv = grad_relu.transpose(0, 2, 1) @ col     # (N, F, 9) -> sum over N
    grad_W_conv = grad_W_conv.sum(axis=0) + L2 * W_conv  # (F, 9)
    grad_b_conv = grad_relu.sum(axis=(0,1))              # (F,)
    # No need to backprop to input
    return grad_W_conv, grad_b_conv, grad_W_lin, grad_b_lin


def train_step(X_batch, y_batch, params, state, t, lr):
    W_conv, b_conv, W_lin, b_lin = params
    m_Wc, m_bc, m_Wl, m_bl = state[0]
    v_Wc, v_bc, v_Wl, v_bl = state[1]
    logits, pool_out, relu_out, col = forward(X_batch, W_conv, b_conv, W_lin, b_lin)
    loss, grad_logits, _ = softmax_cross_entropy(logits, y_batch)
    grads = backward(grad_logits, pool_out, relu_out, col, W_lin, W_conv, X_batch, y_batch, len(y_batch), L2)
    g_Wc, g_bc, g_Wl, g_bl = grads
    # Adam update
    t += 1
    for i, (p, g, m, v) in enumerate(zip(params, grads, [m_Wc, m_bc, m_Wl, m_bl], [v_Wc, v_bc, v_Wl, v_bl])):
        m *= ADAM_BETA1; m += (1 - ADAM_BETA1) * g
        v *= ADAM_BETA2; v += (1 - ADAM_BETA2) * g * g
        m_hat = m / (1 - ADAM_BETA1 ** t)
        v_hat = v / (1 - ADAM_BETA2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
    return loss, t


def evaluate(X, y, params):
    logits, _, _, _ = forward(X, *params)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64
    assert card["classes"] == 10
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(n * 0.8)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Compute normalization on train split
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train_n = normalize(X_train.astype(np.float32), mean, std).reshape(-1, 8, 8, 1)
    X_tune_n = normalize(X_tune.astype(np.float32), mean, std).reshape(-1, 8, 8, 1)

    # Hyperparameter search on tuning split
    best_lr = LR_CANDIDATES[0]
    best_acc = -1.0
    rng = np.random.default_rng(SEED)
    for lr in LR_CANDIDATES:
        # Initialize
        W_conv = (rng.uniform(-1, 1, (FILTERS, 9)) * math.sqrt(6.0 / 9)).astype(np.float32)
        b_conv = np.zeros(FILTERS, np.float32)
        W_lin = (rng.uniform(-1, 1, (CLASSES, FILTERS)) * math.sqrt(6.0 / FILTERS)).astype(np.float32)
        b_lin = np.zeros(CLASSES, np.float32)
        params = [W_conv, b_conv, W_lin, b_lin]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        for epoch in range(EPOCHS_SPLIT):
            order = rng.permutation(len(X_train_n))
            for s in range(0, len(X_train_n), BATCH):
                idx = order[s:s+BATCH]
                loss, t = train_step(X_train_n[idx], y_train[idx], params, state, t, lr)
        acc = evaluate(X_tune_n, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_params = [p.copy() for p in params]

    # Retrain on full train with best_lr
    mean_full = X.mean(axis=0).astype(np.float32)
    std_full = X.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0
    X_full_n = normalize(X.astype(np.float32), mean_full, std_full).reshape(-1, 8, 8, 1)
    y_full = y

    W_conv, b_conv, W_lin, b_lin = best_params
    params = [W_conv, b_conv, W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(EPOCHS_FULL):
        order = rng.permutation(len(X_full_n))
        for s in range(0, len(X_full_n), BATCH):
            idx = order[s:s+BATCH]
            loss, t = train_step(X_full_n[idx], y_full[idx], params, state, t, best_lr)

    # Save weights
    W_conv, b_conv, W_lin, b_lin = params
    out = struct.pack("<I", MAGIC)
    out += struct.pack("<I", FILTERS)
    out += mean_full.astype("<f4").tobytes()
    out += std_full.astype("<f4").tobytes()
    out += W_conv.astype("<f4").tobytes()
    out += b_conv.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Tiny CNN inference for 8x8 optdigits: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel counts 0..16). Output: one byte class 0..9. */
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

#define IN_H 8
#define IN_W 8
#define IN_C 1
#define KERNEL 3
#define PAD 1
#define STRIDE 1
#define OUT_H ((IN_H + 2*PAD - KERNEL) / STRIDE + 1)  // 8
#define OUT_W ((IN_W + 2*PAD - KERNEL) / STRIDE + 1)  // 8
#define POOL_SIZE (OUT_H * OUT_W)  // 64

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 8 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int F = hd[1];  // number of filters
    if (F <= 0 || F > 256) { fprintf(stderr, "bad filter count\n"); return 3; }
    // Expected weight size: magic(4) + F(4) + mean[64]*4 + std[64]*4 + W_conv[F*9]*4 + b_conv[F]*4 + W_lin[10*F]*4 + b_lin[10]*4
    long expected = 8 + 64*4*2 + (long)F*9*4 + (long)F*4 + (long)10*F*4 + 10*4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wb + 8);
    float *mean = ptr; ptr += 64;
    float *std = ptr; ptr += 64;
    float *W_conv = ptr; ptr += (long)F * 9;
    float *b_conv = ptr; ptr += F;
    float *W_lin = ptr; ptr += (long)10 * F;
    float *b_lin = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_img = malloc(IN_H * IN_W * sizeof(float));
    float *col = malloc(POOL_SIZE * 9 * sizeof(float));
    float *conv_out = malloc(POOL_SIZE * F * sizeof(float));
    float *pool_out = malloc(F * sizeof(float));
    float *logits = malloc(10 * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        // Normalize and reshape to 8x8
        for (int p = 0; p < 64; p++) {
            in_img[p] = (xi[p] - mean[p]) / std[p];
        }
        // im2col
        int col_idx = 0;
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                int h_start = oh * STRIDE - PAD;
                int w_start = ow * STRIDE - PAD;
                for (int kh = 0; kh < KERNEL; kh++) {
                    for (int kw = 0; kw < KERNEL; kw++) {
                        int ih = h_start + kh;
                        int iw = w_start + kw;
                        float val = 0.0f;
                        if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                            val = in_img[ih * IN_W + iw];
                        }
                        col[col_idx++] = val;
                    }
                }
            }
        }
        // Convolution: col (64,9) * W_conv^T (9,F) -> (64,F)
        for (int p = 0; p < POOL_SIZE; p++) {
            for (int f = 0; f < F; f++) {
                float sum = b_conv[f];
                for (int k = 0; k < 9; k++) {
                    sum += col[p * 9 + k] * W_conv[f * 9 + k];
                }
                conv_out[p * F + f] = sum > 0.0f ? sum : 0.0f;  // ReLU
            }
        }
        // Global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int p = 0; p < POOL_SIZE; p++) {
                sum += conv_out[p * F + f];
            }
            pool_out[f] = sum / POOL_SIZE;
        }
        // Linear layer
        for (int c = 0; c < 10; c++) {
            float sum = b_lin[c];
            for (int f = 0; f < F; f++) {
                sum += pool_out[f] * W_lin[c * F + f];
            }
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```