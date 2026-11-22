```hypothesis
Family: label-cnn (tiny 1-conv-layer CNN with global average pooling).
Why: The 8x8 digit images have spatial structure; a small CNN exploits local correlations with few parameters (≈170) and should beat linear (0.9613) while staying faster than label-mlp16. The conv layer (8 filters, 3x3, padding=1) plus a linear head is highly efficient at inference.
Failure signal: If dev accuracy ≤ 0.9613 or inference cost exceeds label-linear by >2×, the family does not meet its goal.
```

```python
"""label-cnn: tiny CNN for 8x8 ordinal pixel inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: per-pixel z-score → 8 3x3 filters (pad=1) → ReLU → global avg pool → linear 10.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x434E4E31, d, C, F, K, pad, stride; float32 mean[d], inv_std[d],
conv_w[F][K*K], conv_b[F], fc_w[F][C], fc_b[C] (all little-endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
FILTERS = 8
K = 3
PAD = 1
STRIDE = 1

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
    d, C = X.shape[1], card["classes"]
    assert d == 64 and C == 10

    rng = np.random.default_rng(SEED)
    # per-pixel normalization
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd
    Xn = (X.astype(np.float32) - mu) * inv_sd  # (n, d)
    n = Xn.shape[0]

    # reshape to (n, 1, 8, 8)
    X_img = Xn.reshape(n, 1, 8, 8)

    # He init
    fan_in_conv = 1 * K * K
    conv_w = (rng.normal(0, math.sqrt(2.0 / fan_in_conv), (FILTERS, 1, K, K))).astype(np.float32)
    conv_b = np.zeros(FILTERS, np.float32)
    fan_in_fc = FILTERS
    fc_w = (rng.normal(0, math.sqrt(2.0 / fan_in_fc), (FILTERS, C))).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # precompute patch indices for im2col
    H_out = W_out = 8  # with pad=1, stride=1, kernel=3
    patches_idx = []
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i, j
            patches_idx.append((h_start, w_start))

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_img[idx]          # (b,1,8,8)
            yy = y[idx]             # (b,)
            b = x.shape[0]

            # --- forward ---
            # pad
            x_pad = np.pad(x, ((0,0),(0,0),(PAD,PAD),(PAD,PAD)), mode='constant')
            # im2col: (b, H_out*W_out, K*K)
            patches = np.zeros((b, H_out*W_out, K*K), np.float32)
            for p, (hi, wi) in enumerate(patches_idx):
                patches[:, p, :] = x_pad[:, 0, hi:hi+K, wi:wi+K].reshape(b, K*K)
            # conv: (b, H_out*W_out, FILTERS)
            conv_out = patches @ conv_w.reshape(FILTERS, K*K).T + conv_b
            conv_out = conv_out.reshape(b, H_out, W_out, FILTERS)
            # ReLU
            relu_out = np.maximum(conv_out, 0)
            # global average pool
            pooled = relu_out.mean(axis=(1,2))  # (b, FILTERS)
            # linear
            logits = pooled @ fc_w + fc_b       # (b, C)
            # softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(b), yy] -= 1.0
            grad_logits /= b

            # --- backward ---
            # fc
            grad_fc_w = pooled.T @ grad_logits
            grad_fc_b = grad_logits.sum(axis=0)
            # pool
            grad_pooled = grad_logits @ fc_w.T  # (b, FILTERS)
            # expand to (b, H_out, W_out, FILTERS)
            grad_relu = grad_pooled[:, None, None, :] / (H_out * W_out)
            # ReLU mask
            grad_relu *= (conv_out > 0)
            # conv
            grad_conv_flat = grad_relu.reshape(b, H_out*W_out, FILTERS)
            grad_conv_w = grad_conv_flat.transpose(2,0,1) @ patches  # (FILTERS, K*K)
            grad_conv_b = grad_conv_flat.sum(axis=(0,1))

            grads = [grad_conv_w.reshape(FILTERS,1,K,K), grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr = gr + L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- export weights ---
    # magic, d, C, FILTERS, K, PAD, STRIDE
    hdr = struct.pack("<7i", 0x434E4E31, d, C, FILTERS, K, PAD, STRIDE)
    # mean[d], inv_std[d]
    body = mu.astype("<f4").tobytes() + inv_sd.astype("<f4").tobytes()
    # conv_w[FILTERS][K*K], conv_b[FILTERS], fc_w[FILTERS][C], fc_b[C]
    body += conv_w.reshape(FILTERS, K*K).astype("<f4").tobytes()
    body += conv_b.astype("<f4").tobytes()
    body += fc_w.astype("<f4").tobytes()
    body += fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8 pixels). Output record: 1 byte class.
 * Weights format (little-endian):
 *   int32 magic=0x434E4E31, d, C, F, K, pad, stride
 *   float32 mean[d], inv_std[d]
 *   float32 conv_w[F][K*K], conv_b[F], fc_w[F][C], fc_b[C] */
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
    if (nw < 28 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4], pad = hd[5], stride = hd[6];
    if (d != 64 || C != 10 || F != 8 || K != 3 || pad != 1 || stride != 1) {
        fprintf(stderr, "unexpected architecture parameters\n"); return 3;
    }
    long expected = 28 + 4L * (2L*d + F*K*K + F + F*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 7);
    const float *inv_std = mean + d;
    const float *conv_w = inv_std + d;
    const float *conv_b = conv_w + F * K * K;
    const float *fc_w = conv_b + F;
    const float *fc_b = fc_w + F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *padded = malloc(10 * 10 * sizeof(float));
    float *conv_out = malloc(8 * 8 * F * sizeof(float));
    float *pooled = malloc(F * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize and pad to 10x10 (index [y][x])
        memset(padded, 0, 100 * sizeof(float));
        for (int y = 0; y < 8; y++) {
            for (int x_ = 0; x_ < 8; x_++) {
                int idx = y * 8 + x_;
                float val = (xi[idx] - mean[idx]) * inv_std[idx];
                padded[(y+1)*10 + (x_+1)] = val;
            }
        }

        // convolution: 8 filters, 3x3, stride 1, pad already done
        for (int f = 0; f < F; f++) {
            const float *fw = conv_w + f * 9;
            float bias = conv_b[f];
            for (int y = 0; y < 8; y++) {
                for (int x_ = 0; x_ < 8; x_++) {
                    float sum = bias;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            float w = fw[ky * 3 + kx];
                            float in_val = padded[(y+ky)*10 + (x_+kx)];
                            sum += w * in_val;
                        }
                    }
                    conv_out[(y*8 + x_)*F + f] = sum > 0 ? sum : 0; // ReLU
                }
            }
        }

        // global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0;
            for (int y = 0; y < 8; y++) {
                for (int x_ = 0; x_ < 8; x_++) {
                    sum += conv_out[(y*8 + x_)*F + f];
                }
            }
            pooled[f] = sum / 64.0f;
        }

        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++) {
                sum += pooled[f] * fc_w[f * C + c];
            }
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