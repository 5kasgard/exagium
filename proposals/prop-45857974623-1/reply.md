```hypothesis
Family: label-cnn1d (1D convolutional net on the 64-pixel sequence).
Why: Optdigits pixels have strong local 2D structure; a 1D CNN with small kernel scans the 8x8 rows sequentially, capturing stroke patterns with far fewer parameters than a dense layer. It should beat label-linear (0.961) and approach label-mlp16 (0.977) at lower inference cost (only 8×3×3 conv weights + 8×10 dense).
Failure signal: dev accuracy ≤ 0.961 or Ir/item ≥ 8000.
```

```python
"""label-cnn1d: 1D CNN for optdigits (8x8 pixels as 64-seq).
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: Conv1D(1->8, kernel=3, padding=1) -> ReLU -> GlobalAvgPool -> Dense(8->10).
Training: Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Deterministic seed=1. Exports fixed-layout binary weights."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CONV_OUT, KERNEL = 8, 3
MAGIC = 0x434E4E31  # "CNN1"

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
    n, d = X.shape
    assert d == 64 and C == 10

    # Normalize to [0,1]
    Xf = (X.astype(np.float32) / 16.0).reshape(n, 1, d)  # (N, 1, 64)

    rng = np.random.default_rng(SEED)
    # He init
    conv_w = (rng.normal(0, math.sqrt(2.0 / (1 * KERNEL)), (CONV_OUT, 1, KERNEL))).astype(np.float32)
    conv_b = np.zeros(CONV_OUT, np.float32)
    dense_w = (rng.normal(0, math.sqrt(2.0 / CONV_OUT), (CONV_OUT, C))).astype(np.float32)
    dense_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, dense_w, dense_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(x):
        # x: (B, 1, 64)
        B = x.shape[0]
        # Conv1D with padding=1
        pad = KERNEL // 2
        xp = np.pad(x, ((0,0),(0,0),(pad,pad)), mode='constant')
        out = np.zeros((B, CONV_OUT, d), np.float32)
        for c in range(CONV_OUT):
            w = conv_w[c, 0]
            b = conv_b[c]
            for i in range(d):
                out[:, c, i] = np.dot(xp[:, 0, i:i+KERNEL], w) + b
        out = np.maximum(out, 0)  # ReLU
        # Global average pool
        gap = out.mean(axis=2)  # (B, CONV_OUT)
        logits = gap @ dense_w + dense_b
        return logits, out, gap

    def backward(g_logits, gap, out_conv, x):
        B = x.shape[0]
        # dense grads
        g_dense_w = gap.T @ g_logits
        g_dense_b = g_logits.sum(0)
        g_gap = g_logits @ dense_w.T  # (B, CONV_OUT)
        # pool grad
        g_out_conv = g_gap[:, :, None] / d  # (B, CONV_OUT, 64)
        # ReLU grad
        g_out_conv[out_conv <= 0] = 0
        # conv grads
        g_conv_w = np.zeros_like(conv_w)
        g_conv_b = np.zeros_like(conv_b)
        pad = KERNEL // 2
        xp = np.pad(x, ((0,0),(0,0),(pad,pad)), mode='constant')
        for c in range(CONV_OUT):
            g_conv_b[c] = g_out_conv[:, c].sum()
            for i in range(d):
                g_conv_w[c, 0] += (xp[:, 0, i:i+KERNEL].T @ g_out_conv[:, c, i]).astype(np.float32)
        return [g_conv_w, g_conv_b, g_dense_w, g_dense_b]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = Xf[idx], y[idx]
            logits, out_conv, gap = forward(xb)
            # Softmax + CE grad
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            g_logits = (probs / len(yb)).astype(np.float32)
            grads = backward(g_logits, gap, out_conv, xb)
            # L2
            for i, p in enumerate(params):
                grads[i] += L2 * p
            # Adam
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            conv_w, conv_b, dense_w, dense_b = params

    # Pack weights: magic, d, CONV_OUT, KERNEL, C; then conv_w, conv_b, dense_w, dense_b (all float32 LE)
    header = struct.pack("<5I", MAGIC, d, CONV_OUT, KERNEL, C)
    body = (conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            dense_w.astype("<f4").tobytes() +
            dense_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn1d infer: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output: 1 byte class 0..9.
 * Weights layout (little-endian):
 *   uint32 magic=0x434E4E31, d=64, CONV_OUT=8, KERNEL=3, C=10
 *   float32 conv_w[CONV_OUT][1][KERNEL]
 *   float32 conv_b[CONV_OUT]
 *   float32 dense_w[CONV_OUT][C]
 *   float32 dense_b[C]
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], CONV_OUT = hd[2], KERNEL = hd[3], C = hd[4];
    if (d != 64 || CONV_OUT != 8 || KERNEL != 3 || C != 10) {
        fprintf(stderr, "unexpected architecture params\n"); return 3;
    }
    size_t expected = 20 + 4*(CONV_OUT*KERNEL + CONV_OUT + CONV_OUT*C + C);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *conv_w = (float *)(hd + 5);
    float *conv_b = conv_w + CONV_OUT * KERNEL;
    float *dense_w = conv_b + CONV_OUT;
    float *dense_b = dense_w + CONV_OUT * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(CONV_OUT * d * sizeof(float));
    float *gap = malloc(CONV_OUT * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize input to [0,1] in a local buffer
        float in[64];
        for (int j = 0; j < d; j++) in[j] = xi[j] / 16.0f;

        // Conv1D with padding=1
        for (int c = 0; c < CONV_OUT; c++) {
            const float *w = conv_w + c * KERNEL;
            float b = conv_b[c];
            for (int t = 0; t < d; t++) {
                float sum = b;
                for (int kk = 0; kk < KERNEL; kk++) {
                    int idx = t + kk - 1; // padding=1
                    if (idx >= 0 && idx < d) sum += in[idx] * w[kk];
                }
                conv_out[c * d + t] = sum > 0 ? sum : 0; // ReLU
            }
        }

        // Global average pool
        for (int c = 0; c < CONV_OUT; c++) {
            float sum = 0;
            for (int t = 0; t < d; t++) sum += conv_out[c * d + t];
            gap[c] = sum / d;
        }

        // Dense
        for (int cl = 0; cl < C; cl++) {
            float sum = dense_b[cl];
            for (int c = 0; c < CONV_OUT; c++) sum += gap[c] * dense_w[c * C + cl];
            logits[cl] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int cl = 1; cl < C; cl++) if (logits[cl] > logits[best]) best = (unsigned char)cl;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```