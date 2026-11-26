```hypothesis
Family: label-emb (position-specific embeddings + one hidden ReLU layer).
Why: The 16-byte context benefits from modeling per-position character similarities (embeddings) and non-linear interactions (hidden layer), which linear models miss. Small embedding size (E=8) and hidden layer (H=32) keep inference fast (~5k FLOPs/item) while adding capacity.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost (Ir/item) exceeds label-linear by >2x.
```

```python
"""label-emb: position-specific embeddings + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 8          # embedding dimension per position
H = 32         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50

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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float32)
    # embeddings: [d, 256, E]
    embeddings = (rng.uniform(-0.1, 0.1, (d, 256, E))).astype(np.float32)
    # W1: [d*E, H]
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C]
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward_batch(Xb):
        """Xb: (bs, d) uint8 -> returns logits (bs, C), and cache for backward."""
        bs = Xb.shape[0]
        # concat embeddings: (bs, d*E)
        concat = np.zeros((bs, d * E), dtype=np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = embeddings[j, Xb[:, j]]
        # hidden layer
        hidden_pre = concat @ W1 + b1
        hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
        logits = hidden @ W2 + b2
        return logits, concat, hidden, hidden_pre

    def loss_and_grad(logits, yb):
        """Cross-entropy loss and gradient wrt logits."""
        bs = logits.shape[0]
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(bs), yb] -= 1.0
        grad_logits = probs / bs
        return grad_logits.astype(np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, concat, hidden, hidden_pre = forward_batch(Xb)
            g_logits = loss_and_grad(logits, yb)

            # Backprop
            # Output layer
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_hidden = g_logits @ W2.T
            # ReLU
            g_hidden_pre = g_hidden * (hidden_pre > 0)
            # Hidden layer
            g_W1 = concat.T @ g_hidden_pre + L2 * W1
            g_b1 = g_hidden_pre.sum(axis=0)
            g_concat = g_hidden_pre @ W1.T
            # Embeddings
            g_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                g_embeddings[j, Xb[:, j]] += g_concat[:, j*E:(j+1)*E]

            grads = [g_embeddings, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x454D424C  # 'EMBL' in little-endian bytes
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += embeddings.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic(0x454D424C), d, E, H, C (int32),
 *   embeddings[d][256][E] (float32), W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *embeddings = (const float *)(hd + 5);
    const float *W1 = embeddings + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = (float *)malloc(d * E * sizeof(float));
    float *hidden = (float *)malloc(H * sizeof(float));
    if (!concat || !hidden) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, emb, E * sizeof(float));
        }

        // Hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        int best = 0;
        float best_score = -INFINITY;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            if (sum > best_score) { best_score = sum; best = c; }
        }

        unsigned char out_byte = (unsigned char)best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```