```hypothesis
Family: label-embed (embedding + one hidden ReLU layer). Each of the 16 context bytes is mapped to a 32-dim embedding (position-specific), concatenated to 512 features, then a ReLU layer of 64 units, then 65-class output. This captures byte similarities and positional interactions with ~168k parameters (vs 266k for label-linear) and ~37k multiply-adds per item (vs 266k), so it should be both more accurate and faster. Failure: dev accuracy ≤ 0.34 or instructions/item ≥ label-linear.
```

```python
"""Embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on last 20% of train.in)
E = 32          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
CLIP = 1.0      # gradient clipping

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # W1: (d*E, H), b1: (H,)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C), b2: (C,)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_steps = MAX_EPOCHS * steps_per_epoch

    for step in range(max_steps):
        if step % steps_per_epoch == 0:
            order = rng.permutation(n)
        batch_idx = order[(step % steps_per_epoch) * BATCH : (step % steps_per_epoch + 1) * BATCH]
        xb = X[batch_idx]          # (B, d)
        yb = y[batch_idx]          # (B,)
        B = xb.shape[0]

        # Forward
        # Embedding lookup: (B, d, E)
        emb_x = emb[np.arange(d), xb.T].transpose(1, 0, 2)   # (B, d, E)
        flat = emb_x.reshape(B, d * E)                       # (B, d*E)
        hidden_pre = flat @ W1 + b1                          # (B, H)
        hidden = np.maximum(hidden_pre, 0, dtype=np.float32) # ReLU
        logits = hidden @ W2 + b2                            # (B, C)

        # Loss gradient
        g_logits = softmax_cross_entropy_grad(logits, yb)    # (B, C)

        # Backward
        g_W2 = hidden.T @ g_logits                           # (H, C)
        g_b2 = g_logits.sum(axis=0)                          # (C,)
        g_hidden = g_logits @ W2.T                           # (B, H)
        g_hidden[hidden_pre <= 0] = 0                        # ReLU grad
        g_W1 = flat.T @ g_hidden                             # (d*E, H)
        g_b1 = g_hidden.sum(axis=0)                          # (H,)
        g_flat = g_hidden @ W1.T                             # (B, d*E)
        g_emb_x = g_flat.reshape(B, d, E)                    # (B, d, E)

        # Accumulate embedding gradients
        g_emb = np.zeros_like(emb)
        for j in range(d):
            np.add.at(g_emb[j], xb[:, j], g_emb_x[:, j])

        grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

        # L2 regularization (excluding biases)
        grads[0] += L2 * emb
        grads[1] += L2 * W1
        grads[3] += L2 * W2

        # Gradient clipping
        total_norm = 0.0
        for g in grads:
            total_norm += np.sum(g * g)
        total_norm = math.sqrt(total_norm)
        if total_norm > CLIP:
            scale = CLIP / total_norm
            for g in grads:
                g *= scale

        # Adam update
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * gr
            v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Magic: 0x454D4244 ("EMBD" little-endian), d, E, H, C
    header = struct.pack("<5i", 0x454D4244, d, E, H, C)
    # Flatten in order: emb (d,256,E), W1 (d*E,H), b1 (H), W2 (H,C), b2 (C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding + one hidden ReLU layer for next-byte prediction.
 * ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t b1_sz = H;
    size_t W2_sz = (size_t)H * C;
    size_t b2_sz = C;
    long expected = 20 + 4L * (emb_sz + W1_sz + b1_sz + W2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    float *emb = (float *)(wb + 20);
    float *W1 = emb + emb_sz;
    float *b1 = W1 + W1_sz;
    float *W2 = b1 + b1_sz;
    float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(size_t)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(flat); free(hidden); free(scores);
    free(wb); if (x) free(x);
    return 0;
}
```