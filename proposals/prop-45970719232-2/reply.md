```hypothesis
Family: positional-embedding MLP (pemb-mlp). Each of the 16 context positions has a separate 256×E embedding table; embeddings are summed, passed through one ReLU hidden layer of size H, then a linear output layer to 65 classes. This captures positional information and nonlinear interactions with far fewer parameters than full one-hot linear (label-linear) and should exceed its accuracy. If dev accuracy ≤ 0.3375 or inference cost > label-linear, the hypothesis fails.
```
```python
"""Positional-embedding MLP for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
LR = 1e-3       # Adam learning rate
L2 = 1e-4       # L2 weight decay
BATCH = 64      # minibatch size
EPOCHS = 50     # training epochs
SEED = 1        # fixed seed for reproducibility

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float32)
    embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)   # (16, 256, E)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embed, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Precompute position indices for embedding lookup
    pos_idx = np.arange(d, dtype=np.int32)  # [0,1,...,15]

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, 16) uint8
            yb = y[idx]          # (B,) int64
            B = xb.shape[0]

            # ----- Forward -----
            # Embeddings: (B, 16, E)
            emb = embed[pos_idx, xb]          # advanced indexing
            # Sum over positions -> (B, E)
            h = emb.sum(axis=1)
            # Hidden layer with ReLU
            h = h @ W1 + b1
            h_relu = np.maximum(h, 0)
            # Output logits
            logits = h_relu @ W2 + b2          # (B, C)

            # ----- Loss & gradients -----
            # Softmax cross-entropy
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient of loss w.r.t logits
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            # Output layer
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T

            # ReLU
            grad_h = grad_h_relu * (h > 0)

            # Hidden layer
            grad_W1 = h.T @ grad_h + L2 * W1   # h is (B, E) before ReLU? Wait: h = sum_emb @ W1 + b1, so sum_emb is (B, E)
            # Actually h = sum_emb @ W1 + b1, so grad_W1 = sum_emb.T @ grad_h
            sum_emb = emb.sum(axis=1)          # (B, E)
            grad_W1 = sum_emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            # Embeddings
            grad_emb = np.zeros_like(embed)
            # grad_h flows back to sum_emb: grad_sum_emb = grad_h @ W1.T  (B, E)
            grad_sum_emb = grad_h @ W1.T        # (B, E)
            # Each position gets the same grad_sum_emb (since sum)
            # We need to scatter-add to grad_emb[pos, byte, :]
            # For each position j, for each example i, byte = xb[i,j], add grad_sum_emb[i] to grad_emb[j, byte]
            # Vectorized:
            for j in range(d):
                bytes_j = xb[:, j]             # (B,)
                np.add.at(grad_emb[j], bytes_j, grad_sum_emb)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # ----- Export weights -----
    # Format: magic, d, E, H, C (all int32), then flat float32 arrays in order:
    # embed (d*256*E), W1 (E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x4D4C5031  # "MLP1"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten in C order (row-major)
    embed_flat = embed.astype("<f4").ravel()
    W1_flat = W1.astype("<f4").ravel()
    b1_flat = b1.astype("<f4")
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4")
    out = header + embed_flat.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Positional-embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4D4C5031) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected weight size in bytes
    long expected = 20 +
        (long)d * 256 * E * 4 +   // embed
        (long)E * H * 4 +         // W1
        (long)H * 4 +             // b1
        (long)H * C * 4 +         // W2
        (long)C * 4;              // b2
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    float *p = (float *)(wb + 20);
    float *embed = p; p += (long)d * 256 * E;
    float *W1 = p; p += (long)E * H;
    float *b1 = p; p += H;
    float *W2 = p; p += (long)H * C;
    float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int e = 0; e < E; e++) sum_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            const float *emb_ptr = embed + ((long)j * 256 + byte) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += emb_ptr[e];
        }

        // Hidden layer: ReLU(sum_emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float val = b1[h];
            for (int e = 0; e < E; e++) val += sum_emb[e] * W1[(long)e * H + h];
            hidden[h] = val > 0.0f ? val : 0.0f;
        }

        // Output logits
        for (int c = 0; c < C; c++) {
            float val = b2[c];
            for (int h = 0; h < H; h++) val += hidden[h] * W2[(long)h * C + c];
            logits[c] = val;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(hidden); free(logits);
    free(wb); if (x) free(x);
    return 0;
}
```