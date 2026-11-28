```hypothesis
Family: label-tinycnn — a tiny CNN (1×3×3 conv, 8 filters, ReLU, global avg-pool, linear) for 8×8 digit images.
Why: Optdigits has strong local spatial structure (strokes, loops) that 3×3 filters capture; global avg-pool gives translation invariance with only ~170 params. Should beat label-linear (0.9613) by modeling 2D topology while staying faster than label-mlp16.
Failure signal: Dev accuracy ≤ 0.9613 or Ir/item ≥ 6000 would show the 8-filter design is too weak or too slow.
```

```python
"""label-tinycnn: Tiny CNN for 8x8 ordinal images (optdigits).
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: Input(1,8,8) -> Conv2d(8,3x3,valid) -> ReLU -> GlobalAvgPool -> Linear(10) -> Softmax.
Deterministic (seed=1), Adam lr=1e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  magic=0x434E4E31 ('CNN1'), d=64, C=10, conv_C=8, conv_K=3
  float32: in_mean, in_std
  float32: conv_W[conv_C, conv_K, conv_K], conv_b[conv_C]
  float32: fc_W[C, conv_C], fc_b[C]
"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100
CONV_C, CONV_K = 8, 3  # 8 filters, 3x3 kernel

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
    assert d == 64 and CONV_K == 3
    n = X.shape[0]

    # Global normalization (all pixels ordinal 0..16)
    Xf = X.astype(np.float32)
    in_mean = Xf.mean()
    in_std = Xf.std() or 1.0
    Xn = (Xf - in_mean) / in_std  # (n, 64)
    Xn = Xn.reshape(n, 1, 8, 8)   # NCHW

    rng = np.random.default_rng(SEED)
    # He init for ReLU
    fan_in = CONV_K * CONV_K
    conv_W = (rng.normal(0, math.sqrt(2.0 / fan_in), (CONV_C, 1, CONV_K, CONV_K)).astype(np.float32))
    conv_b = np.zeros(CONV_C, np.float32)
    fc_W = (rng.normal(0, math.sqrt(2.0 / CONV_C), (C, CONV_C)).astype(np.float32))
    fc_b = np.zeros(C, np.float32)

    params = [conv_W, conv_b, fc_W, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(x):
        # x: (B, 1, 8, 8)
        B = x.shape[0]
        # Conv valid 3x3 -> (B, CONV_C, 6, 6)
        out_h = out_w = 8 - CONV_K + 1  # 6
        conv_out = np.zeros((B, CONV_C, out_h, out_w), np.float32)
        for c in range(CONV_C):
            w = conv_W[c, 0]  # (3,3)
            b = conv_b[c]
            for i in range(out_h):
                for j in range(out_w):
                    patch = x[:, 0, i:i+CONV_K, j:j+CONV_K]  # (B,3,3)
                    conv_out[:, c, i, j] = (patch * w).sum(axis=(1,2)) + b
        # ReLU
        h = np.maximum(conv_out, 0)
        # GlobalAvgPool -> (B, CONV_C)
        gap = h.mean(axis=(2,3))
        # Linear
        logits = gap @ fc_W.T + fc_b
        return logits, h, gap

    def softmax_grad(logits, y_idx):
        # logits: (B, C), y_idx: (B,)
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_idx] -= 1.0
        return probs / B  # (B, C)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = Xn[idx]
            yb = y[idx]
            logits, h, gap = forward(xb)
            g_logits = softmax_grad(logits, yb)  # (B, C)

            # fc grads
            g_fc_W = g_logits.T @ gap + L2 * fc_W  # (C, CONV_C)
            g_fc_b = g_logits.sum(axis=0)          # (C,)

            # gap grad -> h grad
            g_gap = g_logits @ fc_W  # (B, CONV_C)
            g_h = np.zeros_like(h)
            for c in range(CONV_C):
                g_h[:, c, :, :] = g_gap[:, c, None, None] / (6*6)

            # ReLU grad
            g_h = g_h * (h > 0)

            # Conv grads
            g_conv_W = np.zeros_like(conv_W)
            g_conv_b = np.zeros_like(conv_b)
            for c in range(CONV_C):
                for i in range(6):
                    for j in range(6):
                        patch = xb[:, 0, i:i+3, j:j+3]  # (B,3,3)
                        g_conv_W[c, 0] += (g_h[:, c, i, j][:, None, None] * patch).sum(axis=0)
                        g_conv_b[c] += g_h[:, c, i, j].sum()
            g_conv_W += L2 * conv_W

            grads = [g_conv_W, g_conv_b, g_fc_W, g_fc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, C, conv_C, conv_K
    header = struct.pack("<5i", 0x434E4E31, d, C, CONV_C, CONV_K)
    body = (
        np.array([in_mean, in_std], dtype=np.float32).tobytes() +
        conv_W.astype("<f4").tobytes() + conv_b.astype("<f4").tobytes() +
        fc_W.astype("<f4").tobytes() + fc_b.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-tinycnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 ordinal pixels). Output: 1 byte class (0-9).
 * Weights format (little-endian):
 *   int32 magic=0x434E4E31, d=64, C=10, conv_C=8, conv_K=3
 *   float32 in_mean, in_std
 *   float32 conv_W[conv_C][conv_K][conv_K], conv_b[conv_C]
 *   float32 fc_W[C][conv_C], fc_b[C]
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], conv_C = hd[3], conv_K = hd[4];
    if (d != 64 || C != 10 || conv_C != 8 || conv_K != 3) { fprintf(stderr, "bad weights dims\n"); return 3; }

    const float *fp = (const float *)(hd + 5);
    float in_mean = fp[0], in_std = fp[1];
    const float *conv_W = fp + 2;                              // [conv_C][conv_K][conv_K]
    const float *conv_b = conv_W + conv_C * conv_K * conv_K;   // [conv_C]
    const float *fc_W = conv_b + conv_C;                       // [C][conv_C]
    const float *fc_b = fc_W + (long)C * conv_C;               // [C]

    long expected = 20 + 4L * (2 + conv_C * conv_K * conv_K + conv_C + (long)C * conv_C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float in_patch[3][3];
    float conv_out[8][6][6];
    float gap[8];
    float logits[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and arrange as 8x8
        float img[8][8];
        for (int p = 0; p < 64; p++) img[p / 8][p % 8] = (xi[p] - in_mean) / in_std;

        // Conv 3x3 valid -> 6x6 per filter
        for (int c = 0; c < conv_C; c++) {
            const float *w = conv_W + c * 9;
            for (int y = 0; y < 6; y++) {
                for (int x_ = 0; x_ < 6; x_++) {
                    float sum = conv_b[c];
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            sum += img[y + ky][x_ + kx] * w[ky * 3 + kx];
                        }
                    }
                    conv_out[c][y][x_] = sum > 0.0f ? sum : 0.0f; // ReLU
                }
            }
        }

        // Global average pool
        for (int c = 0; c < conv_C; c++) {
            float sum = 0.0f;
            for (int y = 0; y < 6; y++)
                for (int x_ = 0; x_ < 6; x_++)
                    sum += conv_out[c][y][x_];
            gap[c] = sum / 36.0f;
        }

        // Linear
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            const float *w = fc_W + (long)c * conv_C;
            for (int u = 0; u < conv_C; u++) sum += gap[u] * w[u];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }
    return 0;
}
```