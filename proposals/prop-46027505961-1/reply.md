```hypothesis
Family: label-tinycnn. A tiny CNN (one 3x3 conv layer with 4 filters, ReLU, global average pooling, linear) exploits the 8x8 spatial structure of optdigits. With only ~200 parameters it should beat the linear baseline (0.9613) at similar or lower inference cost because convolutions are cheap on 8x8 images. Failure: dev accuracy ≤ 0.9613 or instructions/item ≥ label-linear (5058).
```

```python
"""label-tinycnn: one 3x3 conv layer (4 filters), ReLU, global average pooling, linear to 10 classes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only. Exports weights as:
magic 0x54494E43, d=64, H=4, C=10, k=3; float32 mean[d], std[d], conv_w[H][k][k], conv_b[H], lin_w[H][C], lin_b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H, K = 4, 3  # filters, kernel size
C = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    assert d == 64
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # per-pixel z-score
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32).reshape(n, 8, 8, 1)

    # He init for conv (fan_in = 9)
    conv_w = (rng.normal(0, math.sqrt(2.0 / 9.0), (H, K, K)).astype(np.float32))
    conv_b = np.zeros(H, np.float32)
    # He init for linear (fan_in = H)
    lin_w = (rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32))
    lin_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, lin_w, lin_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(x):
        # x: (B,8,8,1)
        # conv: (B,8,8,H) with padding 1
        B = x.shape[0]
        out = np.zeros((B, 8, 8, H), np.float32)
        for f in range(H):
            for i in range(8):
                for j in range(8):
                    s = conv_b[f]
                    for di in range(K):
                        for dj in range(K):
                            ii, jj = i + di - 1, j + dj - 1
                            if 0 <= ii < 8 and 0 <= jj < 8:
                                s += x[b, ii, jj, 0] * conv_w[f, di, dj]
                    out[b, i, j, f] = s
        # ReLU
        out = np.maximum(out, 0)
        # global average pool -> (B, H)
        gap = out.mean(axis=(1, 2))
        # linear
        logits = gap @ lin_w + lin_b
        return logits, gap

    def softmax_grad(logits, y_idx):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_idx] -= 1.0
        return probs / B

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xn[idx], y[idx]
            logits, gap = forward(xb)
            g = softmax_grad(logits, yb)  # (B, C)
            # grad linear
            glin_w = gap.T @ g + L2 * lin_w
            glin_b = g.sum(axis=0)
            # grad gap
            g_gap = g @ lin_w.T  # (B, H)
            # grad ReLU
            g_relu = g_gap[:, None, None, :] * (out > 0)  # (B,8,8,H)
            # grad conv
            gconv_w = np.zeros_like(conv_w)
            gconv_b = np.zeros_like(conv_b)
            for b in range(BATCH):
                for f in range(H):
                    gconv_b[f] += g_relu[b, :, :, f].sum()
                    for di in range(K):
                        for dj in range(K):
                            ii, jj = di - 1, dj - 1
                            if 0 <= ii < 8 and 0 <= jj < 8:
                                gconv_w[f, di, dj] += (xb[b, ii, jj, 0] * g_relu[b, :, :, f]).sum()
            gconv_w += L2 * conv_w

            grads = [gconv_w, gconv_b, glin_w, glin_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # export
    out = struct.pack("<5i", 0x54494E43, d, H, C, K)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += lin_w.astype("<f4").tobytes()
    out += lin_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tinycnn infer: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8). Output: one byte class per record. */
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
    if (nw < 20 || hd[0] != 0x54494E43) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], K = hd[4];
    if (d != 64 || H != 4 || C != 10 || K != 3) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20 + 4L * (2*d + H*K*K + H + H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *mu = p; p += d;
    const float *sd = p; p += d;
    const float *conv_w = p; p += H*K*K;
    const float *conv_b = p; p += H;
    const float *lin_w = p; p += H*C;
    const float *lin_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float in_f[64];
    float conv_out[8][8][4];
    float gap[4];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        // normalize
        for (int j = 0; j < 64; j++) in_f[j] = (xi[j] - mu[j]) / sd[j];
        // conv + ReLU
        for (int f = 0; f < H; f++) {
            for (int ii = 0; ii < 8; ii++) {
                for (int jj = 0; jj < 8; jj++) {
                    float s = conv_b[f];
                    for (int di = 0; di < K; di++) {
                        for (int dj = 0; dj < K; dj++) {
                            int iii = ii + di - 1;
                            int jjj = jj + dj - 1;
                            if (iii >= 0 && iii < 8 && jjj >= 0 && jjj < 8) {
                                s += in_f[iii*8 + jjj] * conv_w[f*K*K + di*K + dj];
                            }
                        }
                    }
                    conv_out[ii][jj][f] = s > 0 ? s : 0;
                }
            }
        }
        // global average pool
        for (int f = 0; f < H; f++) {
            float sum = 0;
            for (int ii = 0; ii < 8; ii++)
                for (int jj = 0; jj < 8; jj++)
                    sum += conv_out[ii][jj][f];
            gap[f] = sum / 64.0f;
        }
        // linear
        for (int c = 0; c < C; c++) {
            float s = lin_b[c];
            for (int f = 0; f < H; f++) s += gap[f] * lin_w[f*C + c];
            logits[c] = s;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```