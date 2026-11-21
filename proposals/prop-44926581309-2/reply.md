```hypothesis
Family: label-embmlp (per-position byte embeddings + one ReLU layer).
Why: The linear model (label-linear) uses one-hot vectors per position, summing them into a 65-d logit. This cannot capture interactions between positions. label-mlp16 adds a hidden layer but keeps the high-dimensional one-hot input (4096), forcing a narrow hidden layer (16). By learning dense embeddings (E=16) per position, we reduce input dimension to 256, allowing a wider hidden layer (H=32) to model position interactions. This should improve accuracy over label-linear (0.3375) while keeping weights smaller (~300KB vs ~1MB) and inference fast.
Failure: If dev accuracy ≤ 0.3375 or inference cost (ns/item) exceeds label-linear by >2x, the model does not meet the goal.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning on the tuning split)
SEED = 1
E = 16          # embedding dimension per position
H = 32          # hidden layer size
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    X_bytes = b"".join(records(sys.argv[2]))
    d = card["input"]["bytes"]
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    embeddings = (rng.normal(0, 0.01, size=(d, 256, E))).astype(np.float32)
    # Initialize W1: (d*E, H) with He initialization
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Initialize W2: (H, C) with He initialization
    W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Flatten embeddings for efficient lookup: (d*256, E)
    emb_flat = embeddings.reshape(-1, E)
    # Position offsets for flat indexing
    pos_offsets = np.arange(d) * 256  # (d,)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]  # (batch, d)
            y_batch = y[idx]
            batch_size = len(idx)

            # Build input: concatenate embeddings
            input_vec = np.empty((batch_size, d * E), dtype=np.float32)
            for j in range(d):
                indices = x_batch[:, j].astype(np.int32) + pos_offsets[j]
                input_vec[:, j * E:(j + 1) * E] = emb_flat[indices]

            # Forward pass
            hidden_pre = input_vec @ W1 + b1  # (batch, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2  # (batch, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(batch_size), y_batch] -= 1.0
            grad_logits /= batch_size

            # Backpropagation
            grad_W2 = hidden.T @ grad_logits  # (H, C)
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T  # (batch, H)
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = input_vec.T @ grad_hidden_pre  # (d*E, H)
            grad_b1 = grad_hidden_pre.sum(axis=0)
            grad_input = grad_hidden_pre @ W1.T  # (batch, d*E)

            # Gradient w.r.t embeddings (accumulate via flat indices)
            grad_emb_flat = np.zeros_like(emb_flat)
            for j in range(d):
                indices = x_batch[:, j].astype(np.int32) + pos_offsets[j]
                np.add.at(grad_emb_flat, indices, grad_input[:, j * E:(j + 1) * E])
            grad_embeddings = grad_emb_flat.reshape(d, 256, E)

            # L2 regularization on weights (not biases)
            grad_embeddings += L2 * embeddings
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Header: magic, d, E, H, C (all int32 little-endian)
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Data in order: embeddings, W1, b1, W2, b2 (all float32 little-endian)
    data = (
        embeddings.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + data)


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
    unsigned char *weights = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)weights;
    if (hd[0] != 0x454D424D) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights: invalid dimensions\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *embeddings = (float *)(weights + 20);
    float *W1 = embeddings + d * 256 * E;
    float *b1 = W1 + d * E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *input_vec = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input_vec by concatenating embeddings
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + (j * 256 + xi[j]) * E;
            memcpy(input_vec + j * E, emb, E * sizeof(float));
        }

        // hidden = ReLU(input_vec * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) {
                sum += input_vec[k] * W1[k * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```