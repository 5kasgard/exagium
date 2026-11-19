```hypothesis
Family: label-embedmlp — per-position byte embeddings (E=32) + one ReLU layer (H=128) + linear output.
Why: The 16 categorical bytes have strong positional and semantic structure; dense embeddings capture character similarities (e.g., vowels cluster) that one-hot cannot, while reducing the hidden-layer input from 4096 to 512 dimensions. A small MLP on top can model non-linear interactions at similar FLOPs to label-linear (≈70k MACs vs 266k).
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference cost > 10k Ir/item.
```

```python
"""label-embedmlp: per-position byte embeddings + 1 ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exported weights.bin format (little-endian):
  int32 magic=0x454D4244, d=16, E=32, H=128, C=65
  float32 emb[d][256][E]       (16*256*32 = 131072 floats)
  float32 W1[d*E][H]           (512*128 = 65536 floats)
  float32 b1[H]                (128 floats)
  float32 W2[H][C]             (128*65 = 8320 floats)
  float32 b2[C]                (65 floats)
Total floats = 131072+65536+128+8320+65 = 205121 → 820484 bytes + 20 header = 820504 bytes.
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)   # (d, 256, E)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p, np.float32), np.zeros_like(p, np.float32)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # Embedding lookup: (B, d, E)
            emb_lookup = emb[np.arange(d), xb.T].transpose(1, 0, 2)   # (B, d, E)
            z = emb_lookup.reshape(B, d * E)                          # (B, d*E)
            a1 = z @ W1 + b1                                          # (B, H)
            h = np.maximum(a1, 0)                                     # ReLU
            logits = h @ W2 + b2                                      # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                                                # (B, C)

            # Backward
            # Output layer
            grad_W2 = h.T @ probs + L2 * W2                           # (H, C)
            grad_b2 = probs.sum(axis=0)                               # (C,)
            grad_h = probs @ W2.T                                     # (B, H)

            # ReLU
            grad_a1 = grad_h * (a1 > 0)                               # (B, H)

            # Hidden layer
            grad_W1 = z.T @ grad_a1 + L2 * W1                         # (d*E, H)
            grad_b1 = grad_a1.sum(axis=0)                             # (H,)
            grad_z = grad_a1 @ W1.T                                   # (B, d*E)

            # Embeddings: grad_z reshaped to (B, d, E)
            grad_emb_lookup = grad_z.reshape(B, d, E)                 # (B, d, E)
            # Accumulate gradients for each (position, byte) pair
            grad_emb = np.zeros_like(emb)
            # For each position j, we need to add grad_emb_lookup[:, j, :] to grad_emb[j, xb[:, j], :]
            # Vectorized via np.add.at
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_lookup[:, j, :])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights.bin
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten in order: emb (d,256,E), W1 (d*E,H), b1 (H), W2 (H,C), b2 (C)
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
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D4244, d, E, H, C
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H]
 *   float32 b1[H]
 *   float32 W2[H][C]
 *   float32 b2[C]
 */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 32 || H != 128 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    // Expected file size: 20 + 4*(d*256*E + d*E*H + H + H*C + C)
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Allocate buffers
    float *z = (float *)malloc(sizeof(float) * d * E);      // concatenated embeddings
    float *h = (float *)malloc(sizeof(float) * H);          // hidden activations
    float *logits = (float *)malloc(sizeof(float) * C);     // output logits

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(z + j * E, e, E * sizeof(float));
        }

        // Hidden layer: h = relu(z @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) {
                sum += z[v] * W1[(long)v * H + u];
            }
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h[u] * W2[(long)u * C + c];
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

    free(z); free(h); free(logits);
    return 0;
}
```