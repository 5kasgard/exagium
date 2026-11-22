```hypothesis
Family: label-mlp8 (one hidden layer of 8 ReLU units). The 8x8 optdigits pixels are ordinal and spatially correlated; a tiny MLP can capture simple non-linear interactions (e.g., strokes) that linear models miss, while staying extremely fast and small. With He initialization, Adam, and per-pixel z-scoring, it should exceed label-linear's 0.9613 dev accuracy. If dev accuracy ≤ 0.962, the hypothesis fails.
```

```python
"""label-mlp8: one hidden layer of 8 ReLU units for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy+stdlib only.
Encoding: per-byte ordinal z-scoring. Hidden size 8. Adam lr 3e-3, L2 1e-4, batch 64, epochs min(100, ceil(600000/n)).
Weights format (little-endian):
  uint32 magic 0x4D4C5038 ('MLP8')
  int32 d, H, C
  float32 mean[d], std[d]
  float32 W1[d*H], b1[H], W2[H*C], b2[C]
"""
import json, math, struct, sys

import numpy as np

H = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # per-pixel z-scoring
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X.astype(np.float32) - mu) / sd).astype(np.float32)  # (n, d)

    n, k = Z.shape
    # He initialization for ReLU
    W1 = (rng.normal(0, math.sqrt(2.0 / k), (k, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)
            # backward
            g_logits = probs.astype(np.float32)
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_h = g_logits @ W2.T
            g_a = g_h * (a > 0)
            g_W1 = z.T @ g_a + L2 * W1
            g_b1 = g_a.sum(axis=0)
            grads = [g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x4D4C5038  # 'MLP8'
    out = struct.pack("<4i", magic, d, H, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp8 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format (little-endian):
 *   uint32 magic 0x4D4C5038, int32 d, H, C
 *   float32 mean[d], std[d], W1[d*H], b1[H], W2[H*C], b2[C]
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x4D4C5038) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int32_t)hd[1], H = (int32_t)hd[2], C = (int32_t)hd[3];
    if (d != 64 || H != 8 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 16
        + (size_t)d * 4 * 2          // mean, std
        + (size_t)d * H * 4          // W1
        + (size_t)H * 4              // b1
        + (size_t)H * C * 4          // W2
        + (size_t)C * 4;             // b2
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std  = mean + d;
    const float *W1   = std + d;
    const float *b1   = W1 + (size_t)d * H;
    const float *W2   = b1 + H;
    const float *b2   = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden layer: h = relu( (xi - mean)/std @ W1 + b1 )
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < d; j++) {
                float z = (xi[j] - mean[j]) / std[j];
                sum += z * W1[(size_t)j * H + u];
            }
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(size_t)u * C + c];
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