```hypothesis
Family: label-embed-mlp (position-specific byte embeddings summed, then one ReLU hidden layer, then linear output).
Why: Learned dense embeddings capture byte similarities and nonlinear context interactions better than one-hot linear models, while the summed embedding keeps the first layer small (E=64) for fast inference. A hidden layer (H=128) adds capacity to model high-order n-gram statistics.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 20000 (≈3× linear cost).
```

```python
"""label-embed-mlp: Position-specific embeddings + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x454D424C, d, E, H, C; then float32 arrays:
  emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (all row-major)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
E = 64          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
CLIP_GRAD = 1.0

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t. logits (already averaged over batch)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    # W1: (E, H)
    W1 = rng.normal(0, math.sqrt(2.0 / E), size=(E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2: (H, C)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yb = y[idx]                # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup and sum: (B, E)
            h0 = np.zeros((B, E), dtype=np.float32)
            for j in range(d):
                h0 += emb[j, xb[:, j], :]   # (B, E)

            # Hidden layer
            h1_pre = h0 @ W1 + b1           # (B, H)
            h1 = np.maximum(h1_pre, 0)      # ReLU
            # Output
            logits = h1 @ W2 + b2           # (B, C)

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)   # (B, C)

            # Backward
            # Output layer
            g_W2 = h1.T @ g_logits + L2 * W2   # (H, C)
            g_b2 = g_logits.sum(axis=0)        # (C,)
            g_h1 = g_logits @ W2.T             # (B, H)
            # ReLU
            g_h1_pre = g_h1 * (h1_pre > 0)     # (B, H)
            # Hidden layer
            g_W1 = h0.T @ g_h1_pre + L2 * W1   # (E, H)
            g_b1 = g_h1_pre.sum(axis=0)        # (H,)
            g_h0 = g_h1_pre @ W1.T             # (B, E)
            # Embeddings
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, scatter-add g_h0 to the corresponding embedding rows
                np.add.at(g_emb[j], xb[:, j], g_h0)
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            # Gradient clipping (global norm)
            total_norm = math.sqrt(sum(np.sum(g*g) for g in grads))
            if total_norm > CLIP_GRAD:
                scale = CLIP_GRAD / total_norm
                grads = [g * scale for g in grads]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    # Header: magic, d, E, H, C (int32)
    header = struct.pack("<5i", 0x454D424C, d, E, H, C)
    # Flatten arrays in order: emb (d,256,E), W1 (E,H), b1 (H), W2 (H,C), b2 (C)
    out_bytes = header
    out_bytes += emb.astype("<f4").tobytes()
    out_bytes += W1.astype("<f4").tobytes()
    out_bytes += b1.astype("<f4").tobytes()
    out_bytes += W2.astype("<f4").tobytes()
    out_bytes += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index 0..C-1). */
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
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    // Expected size: header 20 bytes + 4*(d*256*E + E*H + H + H*C + C)
    long expected = 20 + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *output = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero sum_emb
        for (int e = 0; e < E; e++) sum_emb[e] = 0.0f;
        // Accumulate embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += eptr[e];
        }

        // Hidden layer: h = ReLU(sum_emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int e = 0; e < E; e++) acc += sum_emb[e] * W1[(long)e * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // Output layer: out = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
            output[c] = acc;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(hidden); free(output);
    if (x) free(x);
    free(hd);
    return 0;
}
```