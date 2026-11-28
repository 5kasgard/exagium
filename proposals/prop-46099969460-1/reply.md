```hypothesis
Family: label-mlp-dense. A small MLP (64-32-10) with ReLU on z-scored ordinal pixels.
Why: Nonlinear capacity beats linear (0.9613) with far fewer parameters than per-byte embeddings; 32 hidden units fit the 2867-sample training set without overfitting.
Failure: Dev accuracy ≤0.9613 or inference cost (Ir/item) exceeds label-linear (5059).
```
```python
import json, math, struct, sys

import numpy as np

H = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    n = len(X_bytes)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d).astype(np.float64)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    # Compute mean and std per feature
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    rng = np.random.default_rng(SEED)
    # Initialize weights
    W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_norm[idx].astype(np.float32)
            yy = y[idx]
            # Forward
            h = x @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h[h <= 0] = 0  # ReLU gradient
            grad_W1 = x.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Magic: 'MLPD' little-endian = 0x44504C4D
    magic = 0x44504C4D
    out = struct.pack("<4i", magic, d, H, C)
    out += mean.astype(np.float32).tobytes()
    out += std.astype(np.float32).tobytes()
    out += W1.astype(np.float32).tobytes()
    out += b1.astype(np.float32).tobytes()
    out += W2.astype(np.float32).tobytes()
    out += b2.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44504C4D) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 32 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * (d + d + d*H + H + H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *mean = (float *)(hd + 4);
    float *std = mean + d;
    float *W1 = std + d;
    float *b1 = W1 + (long)d * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d; j++) {
                float val = (xi[j] - mean[j]) / std[j];
                sum += val * W1[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            output[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```