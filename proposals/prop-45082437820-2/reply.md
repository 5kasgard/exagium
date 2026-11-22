```hypothesis
We propose "label-embed-mlp": per-position byte embeddings (dimension 32) concatenated and fed through a two-layer MLP (hidden 128, ReLU). This captures pairwise interactions between context positions that a linear model misses, while keeping inference cost modest (≈70k FLOPs/item). The model should beat label-linear's 0.3375 accuracy on tinyshakespeare next-char prediction. If dev accuracy ≤0.3375 or inference cost exceeds label-gbdt's 131k Ir/item, the approach fails.
```

```python
"""label-embed-mlp: per-position embeddings + 2-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4C31, int32 d,E,H,C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def adam_update(param, grad, m, v, t, lr=LR):
    m *= BETA1; m += (1 - BETA1) * grad
    v *= BETA2; v += (1 - BETA2) * grad * grad
    m_hat = m / (1 - BETA1 ** t)
    v_hat = v / (1 - BETA2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + EPS)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # W1: (d*E, H), He init
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2: (H, C), He init
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    # Adam state
    params = [emb, W1, b1, W2, b2]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: gather embeddings -> (B, d, E) -> (B, d*E)
            emb_batch = emb[np.arange(d), xb.T].transpose(1, 0, 2).reshape(B, -1)  # (B, d*E)
            h_pre = emb_batch @ W1 + b1                      # (B, H)
            h = np.maximum(h_pre, 0)                         # ReLU
            logits = h @ W2 + b2                             # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backward
            grad_W2 = h.T @ grad_logits                      # (H, C)
            grad_b2 = grad_logits.sum(axis=0)                # (C,)

            grad_h = grad_logits @ W2.T                      # (B, H)
            grad_h_pre = grad_h * (h_pre > 0)                # ReLU grad

            grad_W1 = emb_batch.T @ grad_h_pre               # (d*E, H)
            grad_b1 = grad_h_pre.sum(axis=0)                 # (H,)

            grad_emb_flat = grad_h_pre @ W1.T                # (B, d*E)
            grad_emb = grad_emb_flat.reshape(B, d, E)        # (B, d, E)

            # Accumulate embedding gradients per position and byte value
            grad_emb_param = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb_param[j], xb[:, j], grad_emb[:, j, :])

            # L2 regularization
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2
            grad_emb_param += L2 * emb

            # Adam updates
            t += 1
            grads = [grad_emb_param, grad_W1, grad_b1, grad_W2, grad_b2]
            for p, g, (m, v) in zip(params, grads, states):
                adam_update(p, g, m, v, t)

    # Pack weights
    # magic, d, E, H, C
    header = struct.pack("<5i", 0x454D4C31, d, E, H, C)
    # embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W1, b1, W2, b2
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D4C31, int32 d,E,H,C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected weight size: header(20) + embeddings(d*256*E) + W1(d*E*H) + b1(H) + W2(H*C) + b2(C)
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    if (!concat || !hidden || !logits) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```