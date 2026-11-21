```hypothesis
Family: label-embedding-mlp. A two-layer MLP with positional embeddings: each of the 16 input bytes is mapped to a learned D-dimensional embedding (position-specific), the embeddings are summed, then passed through a ReLU hidden layer of size H and a linear output layer of size C=65. This captures non-linear interactions while keeping inference fast (only embedding lookups + two small matvecs). It should beat label-linear (0.3375) on dev accuracy at lower CPU cost because the embedding summation replaces the large one-hot linear layer (4096×65) with a tiny D×H + H×C network. Failure: dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear (7359.9).
```
```python
"""label-embedding-mlp: positional embeddings + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy+stdlib only.
Exports: magic 0x454D4231, d, D, H, C; float32 embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 32        # embedding dimension
H = 64        # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 256
EPOCHS = 20

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
    n = X.shape[0]

    # Tuning split (last 20%) - not used for final training, but we keep the split for reproducibility
    tune_start = int(n * 0.8)
    # We train on all data (as per rules: choose settings on tuning split, then train on all train)

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: [d, 256, D]
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # Hidden layer: W1 [D, H], b1 [H]
    W1 = (rng.normal(0, math.sqrt(2.0 / D), (D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: W2 [H, C], b2 [C]
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Precompute one-hot indices for embedding lookup? We'll just use advanced indexing.
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # Embedding sum: [bs, D]
            emb_sum = np.zeros((bs, D), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]   # (bs, D)

            hidden_pre = emb_sum @ W1 + b1    # (bs, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2         # (bs, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            grad_W2 = hidden.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_hidden = probs @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb_sum = grad_hidden @ W1.T   # (bs, D)

            # Gradient for embeddings: scatter-add
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Export weights
    # Magic, d, D, H, C
    header = struct.pack("<5i", 0x454D4231, d, D, H, C)
    # Embeddings: d * 256 * D floats (row-major: position, byte, dim)
    emb_bytes = emb.astype("<f4").tobytes()
    # W1: D * H, b1: H, W2: H * C, b2: C
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte (class index).
 * Weights format: magic(0x454D4231), d, D, H, C (int32 each), then float32 arrays:
 *   embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C] (all row-major). */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || D <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected weight size: header(20) + embeddings(d*256*D) + W1(D*H) + b1(H) + W2(H*C) + b2(C) floats
    long expected = 20 + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers
    float *emb_sum = (float *)malloc(sizeof(float) * D);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!emb_sum || !hidden || !logits) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding sum
        for (int j = 0; j < D; j++) emb_sum[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) emb_sum[v] += e[v];
        }

        // Hidden layer: ReLU(emb_sum * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int v = 0; v < D; v++) sum += emb_sum[v] * W1[(long)v * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(logits);
    if (x) free(x);
    return 0;
}
```