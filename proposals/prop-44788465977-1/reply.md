```hypothesis
Family: label-mlp (single-hidden-layer ReLU MLP with hidden size tuned on the last 20% of train.in).
Why: Optdigits is a small, low-dimensional vision task where pixel interactions matter; a small MLP captures non-linearities that linear models miss, while staying far cheaper than GBDT or k-NN.
Failure: dev accuracy ≤ 0.9613 (label-linear) or inference cost >2× label-linear.
```

```python
"""label-mlp: single-hidden-layer ReLU MLP for fixed-size byte inputs.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameter (hidden size) chosen on tuning split (last 20% of train.in in file order),
then model retrained on all train. Weights format:
  magic(0x4D4C5031) d H C
  mean[d] std[d] (float32)
  W1[d*H] b1[H] W2[H*C] b2[C] (float32, row-major)"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
HIDDEN_CANDIDATES = [16, 32, 64]  # tuned on validation split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_mlp(X, y, H, C, rng, epochs):
    n, d = X.shape
    # He init
    W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    W1, b1, W2, b2 = params
    a = X @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]

    # z-score per feature on full train (will be saved for inference)
    X = X_bytes.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    n = X.shape[0]
    tune_size = n // 5  # last 20%
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    best_H, best_acc = HIDDEN_CANDIDATES[0], -1.0
    for H in HIDDEN_CANDIDATES:
        params = train_mlp(train_X, train_y, H, C, rng, epochs)
        acc = evaluate(tune_X, tune_y, params)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # retrain on all data with best_H
    final_params = train_mlp(X, y, best_H, C, rng, epochs)
    W1, b1, W2, b2 = final_params

    # write weights.bin
    magic = 0x4D4C5031  # 'MLP1'
    out = struct.pack("<4i", magic, d, best_H, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp infer.c: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format (little-endian):
 *   uint32_t magic(0x4D4C5031), d, H, C
 *   float32 mean[d], std[d]
 *   float32 W1[d*H] (row-major), b1[H], W2[H*C], b2[C] */
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
    uint8_t *wbytes = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wbytes;
    if (hdr[0] != 0x4D4C5031) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], H = hdr[2], C = hdr[3];
    if (d <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 4 * 2          /* mean, std */
        + (size_t)d * H * 4          /* W1 */
        + (size_t)H * 4              /* b1 */
        + (size_t)H * C * 4          /* W2 */
        + (size_t)C * 4;             /* b2 */
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    float *mean = (float *)(wbytes + 16);
    float *std  = mean + d;
    float *W1   = std + d;
    float *b1   = W1 + (size_t)d * H;
    float *W2   = b1 + H;
    float *b2   = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld: %u bytes, not %d\n", i, k, d); return 3; }

        /* input layer: z = (x - mean) / std */
        for (int j = 0; j < d; j++) {
            float z = (xi[j] - mean[j]) / std[j];
            for (int h = 0; h < H; h++) hidden[h] += z * W1[(size_t)j * H + h];
        }
        for (int h = 0; h < H; h++) {
            hidden[h] += b1[h];
            hidden[h] = hidden[h] > 0 ? hidden[h] : 0.0f;  /* ReLU */
        }

        /* output layer */
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            for (int c = 0; c < C; c++) logits[c] += hv * W2[(size_t)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);

        /* reset hidden for next sample */
        memset(hidden, 0, sizeof(float) * H);
    }
    return 0;
}
```