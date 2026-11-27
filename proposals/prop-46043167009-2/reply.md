```hypothesis
Family: label-embmlp (embedding + one hidden ReLU layer).  
Why: The task is character-level language modeling with 16-byte context. A linear model on one-hot bytes (label-linear) ignores positional interactions and caps at ~33.7% accuracy. Learned embeddings per position capture positional n-gram statistics; a hidden ReLU layer adds non-linear combination. This architecture is standard for small context language models and should surpass linear accuracy while remaining fast (≈50k FLOPs/item).  
Failure sign: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2×, the family does not improve the Pareto frontier.
```

```python
"""label-embmlp: embedding + one hidden ReLU layer for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weight format: magic 0x4D4C5031, d, E, H, C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split conceptually)
E = 32          # embedding size per position
H = 256         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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
    # types are all categorical; we use embeddings directly
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)          # (d, 256, E)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute flattened embedding indices for speed: for each position, we will gather rows
    # We'll do embedding lookup by indexing in the loop.

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                     # (B, d)
            yb = y[idx]                     # (B,)

            # Embedding lookup: concatenate embeddings for each position
            # emb shape (d, 256, E) -> for each position j, take emb[j, xb[:, j], :]
            # Result shape (B, d*E)
            B = xb.shape[0]
            x_emb = np.empty((B, d * E), dtype=np.float32)
            for j in range(d):
                x_emb[:, j * E:(j + 1) * E] = emb[j, xb[:, j]]

            # Forward
            h = x_emb @ W1 + b1             # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2       # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = x_emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h[:, j * E:(j + 1) * E])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x4D4C5031, d, E, H, C)
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
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weight format: magic 0x4D4C5031, d, E, H, C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int u = 0; u < H; u++) hidden[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((long)j * E + ee) * H + u];
                hidden[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (hidden[u] < 0.0f) hidden[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = hidden[u];
            for (int c = 0; c < C; c++) logits[c] += hu * W2[(long)u * C + c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```