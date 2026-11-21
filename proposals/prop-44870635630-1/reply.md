```hypothesis
A tiny CNN (label-tinyconv) with 4 3x3 filters exploits 2D spatial structure in 8x8 digits.
It should beat label-linear (96.13%) by capturing local stroke patterns, while staying near 10k Ir/item.
Failure: dev accuracy <= 0.9613 or Ir/item > 15000.
```
```python
"""Tiny CNN for 8x8 digit recognition: 4 3x3 filters -> ReLU -> global avg pool -> linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Exports custom binary format."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
NUM_FILTERS, KERNEL = 4, 3  # 3x3 kernels, 4 filters
POOL_OUT = 6  # 8 - 3 + 1 = 6
MAGIC = 0x434E5631  # 'CNV1'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    assert d == 64

    rng = np.random.default_rng(SEED)

    # Per-pixel normalization (like label-linear)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).astype(np.float32)  # (N, 64)
    X_img = X_norm.reshape(-1, 8, 8)  # (N, 8, 8)

    # Initialize weights
    # conv: (F, 3, 3), conv_b: (F,)
    # fc: (F, C), fc_b: (C,)
    conv_w = (rng.normal(0, math.sqrt(2.0 / (KERNEL * KERNEL)), (NUM_FILTERS, KERNEL, KERNEL)).astype(np.float32))
    conv_b = np.zeros(NUM_FILTERS, np.float32)
    fc_w = (rng.normal(0, math.sqrt(2.0 / NUM_FILTERS), (NUM_FILTERS, C)).astype(np.float32))
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(images):
        # images: (B, 8, 8)
        B = images.shape[0]
        # Conv + ReLU: (B, F, 6, 6)
        conv_out = np.zeros((B, NUM_FILTERS, POOL_OUT, POOL_OUT), np.float32)
        for f in range(NUM_FILTERS):
            for i in range(POOL_OUT):
                for j in range(POOL_OUT):
                    patch = images[:, i:i+KERNEL, j:j+KERNEL]  # (B, 3, 3)
                    conv_out[:, f, i, j] = np.sum(patch * conv_w[f], axis=(1, 2)) + conv_b[f]
        np.maximum(conv_out, 0, out=conv_out)  # ReLU
        # Global average pool: (B, F)
        pooled = conv_out.mean(axis=(2, 3))
        # FC: (B, C)
        logits = pooled @ fc_w + fc_b
        return logits, pooled, conv_out

    def softmax_grad(logits, targets):
        # logits: (B, C), targets: (B,)
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), targets] -= 1.0
        return (probs / B).astype(np.float32)

    t = 0
    n = X_img.shape[0]
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_img[idx], y[idx]
            logits, pooled, conv_out = forward(xb)
            g_logits = softmax_grad(logits, yb)  # (B, C)

            # Gradients
            g_fc_w = pooled.T @ g_logits + L2 * fc_w  # (F, C)
            g_fc_b = g_logits.sum(axis=0)             # (C,)

            # Backprop through pooling: (B, F, 6, 6)
            g_pooled = g_logits @ fc_w.T              # (B, F)
            g_conv_out = g_pooled[:, :, None, None] / (POOL_OUT * POOL_OUT)  # (B, F, 1, 1) broadcast

            # Backprop through ReLU
            g_conv_out = g_conv_out * (conv_out > 0).astype(np.float32)

            # Backprop through conv
            g_conv_w = np.zeros_like(conv_w)
            g_conv_b = np.zeros_like(conv_b)
            for f in range(NUM_FILTERS):
                for i in range(POOL_OUT):
                    for j in range(POOL_OUT):
                        patch = xb[:, i:i+KERNEL, j:j+KERNEL]  # (B, 3, 3)
                        g = g_conv_out[:, f, i, j][:, None, None]  # (B, 1, 1)
                        g_conv_w[f] += np.sum(patch * g, axis=0)
                        g_conv_b[f] += np.sum(g)
            g_conv_w += L2 * conv_w

            grads = [g_conv_w, g_conv_b, g_fc_w, g_fc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # Format: magic, d, H, C, mean[64], std[64], conv_w[F][3][3], conv_b[F], fc_w[F][C], fc_b[C]
    out = struct.pack("<4i", MAGIC, d, NUM_FILTERS, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += fc_w.astype("<f4").tobytes()
    out += fc_b.astype("<f4").tobytes()

    expected = 16 + 4 * (64 * 2 + NUM_FILTERS * KERNEL * KERNEL + NUM_FILTERS + NUM_FILTERS * C + C)
    if len(out) != expected:
        print(f"weight size mismatch: {len(out)} != {expected}", file=sys.stderr)
        sys.exit(1)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Tiny CNN for 8x8 digits: 4 3x3 filters -> ReLU -> global avg pool -> linear.
 * ./infer weights.bin in.rec N
 * Weights: magic 0x434E5631, d=64, H=4, C=10, mean[64], std[64], conv_w[4][3][3], conv_b[4], fc_w[4][10], fc_b[10] */
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
    if (nw < 16 || hd[0] != 0x434E5631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 4 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    const int KERNEL = 3, POOL_OUT = 6;
    long expected = 16 + 4L * (64 * 2 + H * KERNEL * KERNEL + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mean = p; p += 64;
    const float *std = p; p += 64;
    const float *conv_w = p; p += H * KERNEL * KERNEL;
    const float *conv_b = p; p += H;
    const float *fc_w = p; p += H * C;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float norm[64];
    float conv_out[4][6][6];
    float pooled[4];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < 64; j++) {
            norm[j] = (xi[j] - mean[j]) / std[j];
        }

        // Conv + ReLU
        for (int f = 0; f < H; f++) {
            for (int ii = 0; ii < POOL_OUT; ii++) {
                for (int jj = 0; jj < POOL_OUT; jj++) {
                    float sum = conv_b[f];
                    for (int di = 0; di < KERNEL; di++) {
                        for (int dj = 0; dj < KERNEL; dj++) {
                            int idx = (ii + di) * 8 + (jj + dj);
                            int widx = (f * KERNEL + di) * KERNEL + dj;
                            sum += norm[idx] * conv_w[widx];
                        }
                    }
                    conv_out[f][ii][jj] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // Global average pool
        for (int f = 0; f < H; f++) {
            float sum = 0.0f;
            for (int ii = 0; ii < POOL_OUT; ii++)
                for (int jj = 0; jj < POOL_OUT; jj++)
                    sum += conv_out[f][ii][jj];
            pooled[f] = sum / (POOL_OUT * POOL_OUT);
        }

        // FC
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < H; f++)
                sum += pooled[f] * fc_w[f * C + c];
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```