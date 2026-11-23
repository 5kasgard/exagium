```hypothesis
Family: label-tinycnn — a tiny convolutional network (3×3 conv, 8 filters, ReLU, global average pooling, 10-class linear).
Why: 8×8 digit images have spatial structure; a small CNN captures local patterns with far fewer parameters than an MLP and should exceed linear accuracy (0.9613) while staying well below the cost of label-gbdt (41k Ir/item).
Failure: if dev accuracy ≤ 0.9613 or instructions/item ≥ 40000, the model does not justify its complexity.
```

```python
"""label-tinycnn: tiny CNN for 8x8 optdigits.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: input 8x8x1 -> conv3x3 8 filters (pad 1) -> ReLU -> global avg pool -> linear 8->10.
Trains on first 80% of train.in, validates on last 20% (tuning split) to pick epochs,
then retrains on all train.in for that many epochs.
Weights format (little-endian):
  magic 0x544E4E43 ('CNNT'), d=64, H=8, C=10 (int32 each)
  mean[64] float32, inv_std[64] float32
  conv_w[8][3][3] float32, conv_b[8] float32
  fc_w[8][10] float32, fc_b[10] float32
"""
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col(x, kh, kw, pad, stride):
    # x: (N, H, W, C) -> (N*OH*OW, kh*kw*C)
    N, H, W, C = x.shape
    OH = (H + 2*pad - kh) // stride + 1
    OW = (W + 2*pad - kw) // stride + 1
    x_pad = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    strides = x_pad.strides
    shape = (N, OH, OW, kh, kw, C)
    strides_ = (strides[0], strides[1]*stride, strides[2]*stride, strides[1], strides[2], strides[3])
    patches = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides_, writeable=False)
    return patches.reshape(N * OH * OW, kh * kw * C), (N, OH, OW)

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    H, W = 8, 8  # spatial
    assert d == H * W
    n = X_bytes.shape[0]

    # normalization per pixel
    rng = np.random.default_rng(SEED)
    X = X_bytes.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    inv_std = 1.0 / std
    X = (X - mean) * inv_std
    X = X.reshape(-1, H, W, 1)  # (N, 8, 8, 1)

    # split: last 20% for tuning
    split = int(0.8 * n)
    idx = np.arange(n)
    # keep file order for tuning split (last 20% in file order)
    train_idx = idx[:split]
    tune_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    n_train = len(train_idx)

    # initialize weights
    # conv: 8 filters, 3x3, 1 channel -> (9, 8)
    fan_in = 3*3*1
    conv_w = (rng.uniform(-1, 1, (fan_in, 8)) * math.sqrt(6.0 / fan_in)).astype(np.float32)
    conv_b = np.zeros(8, np.float32)
    # fc: 8 -> 10
    fc_w = (rng.uniform(-1, 1, (8, C)) * math.sqrt(6.0 / 8)).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    best_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # shuffle training data
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx_batch = order[s:s+BATCH]
            xb = X_train[idx_batch]
            yb = y_train[idx_batch]
            B = xb.shape[0]

            # forward
            # conv
            patches, (N, OH, OW) = im2col(xb, 3, 3, 1, 1)  # (B*64, 9)
            conv_out = patches @ conv_w + conv_b  # (B*64, 8)
            conv_out = conv_out.reshape(B, OH, OW, 8)  # (B, 8, 8, 8)
            relu_out = np.maximum(conv_out, 0)
            # global avg pool
            pool_out = relu_out.mean(axis=(1,2))  # (B, 8)
            # fc
            logits = pool_out @ fc_w + fc_b  # (B, C)

            # loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # backward fc
            dfc_w = pool_out.T @ dlogits  # (8, C)
            dfc_b = dlogits.sum(axis=0)   # (C,)
            dpool = dlogits @ fc_w.T      # (B, 8)

            # backward pool (avg over 64 positions)
            drelu = dpool[:, None, None, :] / 64.0  # (B, 1, 1, 8) -> broadcast to (B,8,8,8)
            drelu = drelu * (conv_out > 0)  # ReLU grad

            # backward conv
            dconv_out = drelu.reshape(B * OH * OW, 8)
            dconv_w = patches.T @ dconv_out  # (9, 8)
            dconv_b = dconv_out.sum(axis=0)  # (8,)

            grads = [dconv_w, dconv_b, dfc_w, dfc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr += L2 * p  # L2 regularization
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # evaluate on tuning split
        patches_t, _ = im2col(X_tune, 3, 3, 1, 1)
        conv_out_t = patches_t @ conv_w + conv_b
        conv_out_t = conv_out_t.reshape(len(X_tune), 8, 8, 8)
        relu_out_t = np.maximum(conv_out_t, 0)
        pool_out_t = relu_out_t.mean(axis=(1,2))
        logits_t = pool_out_t @ fc_w + fc_b
        pred_t = logits_t.argmax(axis=1)
        acc = (pred_t == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
        elif epoch - best_epoch >= PATIENCE:
            break

    # retrain on all data for best_epoch epochs
    conv_w, conv_b, fc_w, fc_b = best_params
    params = [conv_w, conv_b, fc_w, fc_b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s+BATCH]
            xb = X[idx_batch]
            yb = y[idx_batch]
            B = xb.shape[0]

            patches, _ = im2col(xb, 3, 3, 1, 1)
            conv_out = patches @ conv_w + conv_b
            conv_out = conv_out.reshape(B, 8, 8, 8)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.mean(axis=(1,2))
            logits = pool_out @ fc_w + fc_b

            dlogits = softmax_cross_entropy_grad(logits, yb)

            dfc_w = pool_out.T @ dlogits
            dfc_b = dlogits.sum(axis=0)
            dpool = dlogits @ fc_w.T

            drelu = dpool[:, None, None, :] / 64.0
            drelu = drelu * (conv_out > 0)
            dconv_out = drelu.reshape(B * 64, 8)
            dconv_w = patches.T @ dconv_out
            dconv_b = dconv_out.sum(axis=0)

            grads = [dconv_w, dconv_b, dfc_w, dfc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr += L2 * p
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    conv_w, conv_b, fc_w, fc_b = params
    magic = 0x544E4E43  # 'CNNT'
    header = struct.pack("<4i", magic, d, 8, C)
    body = (
        mean.astype("<f4").tobytes() +
        inv_std.astype("<f4").tobytes() +
        conv_w.astype("<f4").tobytes() +
        conv_b.astype("<f4").tobytes() +
        fc_w.astype("<f4").tobytes() +
        fc_b.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-tinycnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8). Output record: 1 byte class.
 * Weights format (little-endian):
 *   magic 0x544E4E43, d=64, H=8, C=10 (int32)
 *   mean[64] float32, inv_std[64] float32
 *   conv_w[8][3][3] float32, conv_b[8] float32
 *   fc_w[8][10] float32, fc_b[10] float32
 */
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
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x544E4E43) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 8 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (64 + 64 + 8*3*3 + 8 + 8*10 + 10);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    const float *mean = ptr; ptr += 64;
    const float *inv_std = ptr; ptr += 64;
    const float *conv_w = ptr; ptr += 8*3*3;
    const float *conv_b = ptr; ptr += 8;
    const float *fc_w = ptr; ptr += 8*10;
    const float *fc_b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;

    float input[8][8];
    float conv_out[8][8][8];
    float pool[8];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *rec = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // normalize and reshape
        for (int p = 0; p < 64; p++) {
            int y = p / 8, x_ = p % 8;
            input[y][x_] = (rec[p] - mean[p]) * inv_std[p];
        }

        // conv 3x3, 8 filters, pad 1, stride 1
        for (int f = 0; f < 8; f++) {
            const float *w = conv_w + f * 9; // 3x3
            float b = conv_b[f];
            for (int y = 0; y < 8; y++) {
                for (int x_ = 0; x_ < 8; x_++) {
                    float sum = b;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = y + ky - 1;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = x_ + kx - 1;
                            if (iy >= 0 && iy < 8 && ix >= 0 && ix < 8) {
                                sum += input[iy][ix] * w[ky * 3 + kx];
                            }
                        }
                    }
                    conv_out[y][x_][f] = sum > 0.0f ? sum : 0.0f; // ReLU
                }
            }
        }

        // global average pooling
        for (int f = 0; f < 8; f++) {
            float sum = 0.0f;
            for (int y = 0; y < 8; y++)
                for (int x_ = 0; x_ < 8; x_++)
                    sum += conv_out[y][x_][f];
            pool[f] = sum / 64.0f;
        }

        // fully connected
        for (int c = 0; c < 10; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < 8; f++)
                sum += pool[f] * fc_w[f * 10 + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }
    return 0;
}
```