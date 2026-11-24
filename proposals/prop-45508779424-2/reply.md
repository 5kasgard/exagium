```hypothesis
Family: label-embed-mlp. A small MLP with per-position byte embeddings (dim 8) and one ReLU hidden layer (size 64). This captures interactions between positions via the hidden layer, while embeddings reduce parameter count vs one-hot linear. Expect >35% accuracy on dev, beating label-linear (33.75%). If accuracy <=33.75% or inference cost >2x label-linear, the hypothesis fails.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_bytes.shape[1]
    C = card["classes"]
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: (d*E, H)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yb = y[idx]                # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, E)
            emb_lookup = emb[np.arange(d)[None, :], xb]   # (B, d, E)
            flat = emb_lookup.reshape(B, d * E)           # (B, d*E)
            hidden_pre = flat @ W1 + b1                   # (B, H)
            hidden = np.maximum(hidden_pre, 0)            # ReLU
            logits = hidden @ W2 + b2                     # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                       # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2    # (H, C)
            grad_b2 = grad_logits.sum(axis=0)             # (C,)
            grad_hidden = grad_logits @ W2.T              # (B, H)
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)  # ReLU grad
            grad_W1 = flat.T @ grad_hidden_pre + L2 * W1  # (d*E, H)
            grad_b1 = grad_hidden_pre.sum(axis=0)         # (H,)
            grad_flat = grad_hidden_pre @ W1.T            # (B, d*E)
            grad_emb_lookup = grad_flat.reshape(B, d, E)  # (B, d, E)

            # Accumulate embedding gradients
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, (np.arange(d)[None, :], xb), grad_emb_lookup)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C, then embeddings, W1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D4245  # 'EMBE'
    header = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    // Expected weight size: header(20) + emb(d*256*E) + W1(d*E*H) + b1(H) + W2(H*C) + b2(C) all float32
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: flat (d*E) @ W1 (d*E x H) + b1
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f; // ReLU
        }

        // Output layer: hidden (H) @ W2 (H x C) + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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