```hypothesis
Family: label-embmlp — per-position byte embeddings (E=8) concatenated and passed through a 64-unit ReLU hidden layer to 65-class softmax.
Why: The 16-byte context has 4096 one-hot features; dense embeddings capture character similarities (e.g., vowels, punctuation) and reduce parameters, improving generalization on the small (24k) tinyshakespeare training set. A single hidden layer adds non-linearity for context interactions.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference cost (Ir/item) exceeds label-linear by >20%.
```

```python
"""label-embmlp: per-position byte embeddings + 1-hidden-layer MLP for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only.
Weights format: magic 0x454D424C, int32 d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
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
    d = card["input"]["bytes"]
    C = card["classes"]
    # Read data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: [d, 256, E]
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    # Hidden layer: input dim = d*E
    in_dim = d * E
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, targets):
        # logits: (batch, C), targets: (batch,)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return (probs / len(targets)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_idx = X_bytes[idx]          # (batch, d)
            yy = y[idx]                   # (batch,)

            # Embedding lookup: (batch, d, E)
            x_emb = emb[np.arange(d), x_idx.T].transpose(1, 0, 2)  # (batch, d, E)
            x_flat = x_emb.reshape(len(idx), -1)                   # (batch, d*E)

            # Forward
            hidden_pre = x_flat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            # Loss gradient
            g_logits = softmax_grad(logits, yy)   # (batch, C)

            # Backprop output layer
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_hidden = g_logits @ W2.T            # (batch, H)

            # ReLU gradient
            g_hidden_pre = g_hidden * (hidden_pre > 0)

            # Backprop hidden layer
            g_W1 = x_flat.T @ g_hidden_pre + L2 * W1
            g_b1 = g_hidden_pre.sum(axis=0)
            g_x_flat = g_hidden_pre @ W1.T        # (batch, d*E)

            # Backprop embeddings
            g_x_emb = g_x_flat.reshape(len(idx), d, E)   # (batch, d, E)
            g_emb = np.zeros_like(emb)
            # Scatter-add gradients to embedding positions
            for j in range(d):
                # x_idx[:, j] gives indices for position j (batch,)
                # g_x_emb[:, j, :] are gradients for those embeddings
                np.add.at(g_emb[j], x_idx[:, j], g_x_emb[:, j, :])
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            t += 1
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Magic: 0x454D424C ("LBME" little-endian)
    out = struct.pack("<4i", 0x454D424C, d, E, H, C)
    # Embeddings: [d][256][E] in order
    out += emb.astype("<f4").tobytes()
    # W1, b1, W2, b2
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
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic 0x454D424C, int32 d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    int in_dim = d * E;
    long expected = 20
        + 4L * d * 256 * E
        + 4L * in_dim * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + in_dim * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!hidden || !logits) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings into a flat buffer (on stack)
        float x_flat[16 * 8];  // d*E max 128
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            memcpy(x_flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(x_flat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int m = 0; m < in_dim; m++) sum += x_flat[m] * W1[m * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
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
    return 0;
}
```