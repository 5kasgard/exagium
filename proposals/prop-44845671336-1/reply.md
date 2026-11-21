```hypothesis
Family: label-tinycnn — a tiny CNN (3×3 conv, 8 filters, ReLU, global average pooling, linear) for 8×8 images.
Why: Optdigits are 8×8 digit images; a tiny CNN captures local spatial structure with very few parameters (~170) and should exceed linear accuracy (0.9613) while keeping inference cost near linear (only 512 conv MACs + 80 linear MACs per item).
Failure signal: If dev accuracy ≤ 0.9613 or instructions/item exceeds label-linear by >20%, the family does not meet its goal.
```

```python
"""Tiny CNN for 8×8 ordinal images: 3×3 conv (8 filters, pad=1), ReLU, global average pooling, linear.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exported format (little-endian):
  int32 magic=0x54434E31, d=64, C=10, F=8, K=3
  float32 mean[d], std[d]
  float32 conv_w[F][K][K], conv_b[F]
  float32 fc_w[F][C], fc_b[C]"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
F, K = 8, 3  # filters, kernel size


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
    d = X.shape[1]
    assert d == 64 and K == 3 and F == 8

    rng = np.random.default_rng(SEED)

    # per-pixel mean/std
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32).reshape(-1, 1, 8, 8)

    # init weights
    conv_w = (rng.uniform(-1, 1, (F, 1, K, K)) * math.sqrt(6.0 / (K * K))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    fc_w = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Xn.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = Xn[idx]
            yy = y[idx]
            bsz = x.shape[0]

            # forward
            # conv: (bsz,1,8,8) * (F,1,3,3) -> (bsz,F,8,8) with padding=1
            # im2col for 3x3 kernel, stride=1, pad=1
            x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
            cols = np.zeros((bsz, F, 8, 8, K * K), np.float32)
            for i in range(3):
                for j in range(3):
                    cols[:, :, :, :, i * 3 + j] = x_pad[:, :, i:i + 8, j:j + 8]
            conv_out = np.einsum('bfhwk,fk->bfhw', cols, conv_w.reshape(F, K * K)) + conv_b.reshape(1, F, 1, 1)
            h = np.maximum(conv_out, 0)  # ReLU
            # global average pooling
            gap = h.mean(axis=(2, 3))  # (bsz, F)
            logits = gap @ fc_w + fc_b  # (bsz, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad_logits = probs / bsz

            # backward
            grad_fc_w = gap.T @ grad_logits + L2 * fc_w
            grad_fc_b = grad_logits.sum(axis=0)
            grad_gap = grad_logits @ fc_w.T  # (bsz, F)
            grad_h = grad_gap[:, :, None, None] / (8 * 8)  # (bsz, F, 1, 1) broadcast to (bsz,F,8,8)
            grad_h *= (conv_out > 0)  # ReLU grad

            # conv grads
            grad_conv_w = np.einsum('bfhw,bfhwk->fk', grad_h, cols).reshape(F, 1, K, K) + L2 * conv_w
            grad_conv_b = grad_h.sum(axis=(0, 2, 3))

            grads = [grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights
    magic = 0x54434E31  # 'T C N 1'
    out = struct.pack("<5i", magic, d, C, F, K)
    out += mu.tobytes() + sd.tobytes()
    out += conv_w.astype("<f4").tobytes() + conv_b.astype("<f4").tobytes()
    out += fc_w.astype("<f4").tobytes() + fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Tiny CNN inference for 8×8 images: 3×3 conv (8 filters, pad=1), ReLU, global avg pool, linear.
 * ./infer weights.bin in.rec N
 * Input record: 64 bytes. Output record: 1 byte (class 0..9). */
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
    if (nw < 20 || hd[0] != 0x54434E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4];
    if (d != 64 || C != 10 || F != 8 || K != 3) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * (d * 2 + F * K * K + F + F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 5);
    const float *sd = mu + d;
    const float *conv_w = sd + d;
    const float *conv_b = conv_w + F * K * K;
    const float *fc_w = conv_b + F;
    const float *fc_b = fc_w + F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float gap[F];
    float logits[C];
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize and reshape to 8x8 with padding=1 (10x10)
        float img_pad[10][10] = {{0}};
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int idx = r * 8 + c;
                img_pad[r + 1][c + 1] = (xi[idx] - mu[idx]) / sd[idx];
            }
        }

        // convolution + ReLU
        float conv_out[F][8][8];
        for (int f = 0; f < F; f++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = conv_b[f];
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            sum += img_pad[r + kr][c + kc] * conv_w[f * K * K + kr * K + kc];
                        }
                    }
                    conv_out[f][r][c] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    sum += conv_out[f][r][c];
            gap[f] = sum / 64.0f;
        }

        // linear
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++)
                sum += gap[f] * fc_w[f * C + c];
            logits[c] = sum;
        }

        // argmax
        best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }
    return 0;
}
```