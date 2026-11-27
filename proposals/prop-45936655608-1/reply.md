```hypothesis
Family: label-elm (Extreme Learning Machine with fixed random input weights and trained output layer).
Why: The optdigits pixels are ordinal; a random projection with ReLU expands features non-linearly, and a linear output layer can capture complex boundaries. Fixed input weights avoid overfitting and make training fast (only output layer learned). Tuning hidden size on the 20% split should yield >0.96 accuracy with inference cost similar to a small MLP.
Failure: If validation accuracy on tuning split does not exceed label-linear (0.9613), the family does not beat the baseline.
```

```python
"""label-elm: Extreme Learning Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal features by z-scoring. Uses fixed random input weights (Xavier uniform)
with ReLU, then trains output layer (softmax) with Adam (lr=3e-3, L2=1e-4).
Hidden size H chosen on last 20% of train.in (tuning split), then retrained on all data.
Weights format: magic 0x454C4D41, d, H, C; float32 mean[d], std[d], W_in[d*H], W_out[H*C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = [16, 32, 64, 128]  # hidden sizes to try on tuning split
PROJ_SEED_BASE = 12345  # base seed for fixed random projection


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - one_hot) / n."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_output(X, y, H, C, rng):
    """Train output layer weights W_out (H x C) and b_out (C) on features X (n x H)."""
    n = X.shape[0]
    W_out = np.zeros((H, C), np.float32)
    b_out = np.zeros(C, np.float32)
    params = [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits = xb @ W_out + b_out
            g = softmax_grad(logits, yb)
            grad_W = xb.T @ g + L2 * W_out
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W_out, b_out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X_bytes.shape[0]
    # Split: last 20% for tuning (validation)
    split = int(n_total * 0.8)
    X_tr, X_val = X_bytes[:split], X_bytes[split:]
    y_tr, y_val = y[:split], y[split:]

    # Encode ordinal features: z-score per column on training portion
    means = X_tr.mean(axis=0).astype(np.float64)
    stds = X_tr.std(axis=0).astype(np.float64)
    stds[stds == 0] = 1.0
    def encode(X):
        return ((X.astype(np.float64) - means) / stds).astype(np.float32)

    X_tr_enc = encode(X_tr)
    X_val_enc = encode(X_val)

    best_H = H_CANDIDATES[0]
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    for H in H_CANDIDATES:
        # Fixed random input projection
        proj_rng = np.random.default_rng(PROJ_SEED_BASE + H)
        limit = math.sqrt(6.0 / d)
        W_in = proj_rng.uniform(-limit, limit, (d, H)).astype(np.float32)
        # Compute hidden features (ReLU)
        H_tr = np.maximum(X_tr_enc @ W_in, 0, dtype=np.float32)
        H_val = np.maximum(X_val_enc @ W_in, 0, dtype=np.float32)
        # Train output layer
        W_out, b_out = train_output(H_tr, y_tr, H, C, rng)
        # Evaluate
        logits = H_val @ W_out + b_out
        preds = logits.argmax(axis=1)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on full data with best_H
    proj_rng = np.random.default_rng(PROJ_SEED_BASE + best_H)
    limit = math.sqrt(6.0 / d)
    W_in = proj_rng.uniform(-limit, limit, (d, best_H)).astype(np.float32)
    X_all_enc = encode(X_bytes)
    H_all = np.maximum(X_all_enc @ W_in, 0, dtype=np.float32)
    W_out, b_out = train_output(H_all, y, best_H, C, rng)

    # Save weights
    # magic, d, H, C
    header = struct.pack("<4i", 0x454C4D41, d, best_H, C)
    # mean[d], std[d]
    mean_bytes = means.astype("<f4").tobytes()
    std_bytes = stds.astype("<f4").tobytes()
    # W_in[d*H] (row-major: d rows, H cols)
    W_in_bytes = W_in.astype("<f4").tobytes()
    # W_out[H*C] (row-major: H rows, C cols)
    W_out_bytes = W_out.astype("<f4").tobytes()
    # b_out[C]
    b_out_bytes = b_out.astype("<f4").tobytes()

    out = header + mean_bytes + std_bytes + W_in_bytes + W_out_bytes + b_out_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal). Output: one byte class index.
 * Weights format: magic 0x454C4D41, d, H, C; float32 mean[d], std[d], W_in[d*H], W_out[H*C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454C4D41) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected size: 16 + 4*d + 4*d + 4*d*H + 4*H*C + 4*C
    long expected = 16L + 4L*d + 4L*d + 4L*d*H + 4L*H*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W_in = std + d;
    const float *W_out = W_in + (long)d * H;
    const float *b_out = W_out + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!hidden || !scores) { fprintf(stderr, "alloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and compute hidden layer (ReLU)
        for (int h = 0; h < H; h++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                float xn = (xi[j] - mean[j]) / std[j];
                sum += xn * W_in[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W_out[(long)h * C + c];
            }
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores);
    if (x) free(x);
    free(wbuf);
    return 0;
}
```