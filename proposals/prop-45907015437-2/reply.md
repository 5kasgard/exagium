```hypothesis
Family: label-embedmlp – a small MLP with a shared byte embedding across the 16 context positions.
Why: The linear model treats each position independently with one-hot vectors. Sharing a low-dimensional embedding (8-d) across positions captures character similarities and reduces parameters, while a hidden layer (32 ReLU units) models interactions. This should beat label-linear's 0.3375 accuracy at lower inference cost (fewer multiply-adds).
Fail condition: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear, the family is not justified.
```

```python
"""label-embedmlp: shared byte embedding + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format (little-endian):
  int32 magic=0x454D4244 ('EMBD'), d=16, E=8, H=32, C=65
  float32 emb[256][E]
  float32 W1[d*E][H]   (row-major: 16 blocks of E x H)
  float32 b1[H]
  float32 W2[H][C]
  float32 b2[C]
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
E = 8          # embedding dimension
H = 32         # hidden layer size
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
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65

    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, 16) uint8
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, 16, E)
            x_emb = emb[xb]      # (B, 16, E)
            x_flat = x_emb.reshape(B, d * E)   # (B, 128)
            hidden_pre = x_flat @ W1 + b1      # (B, H)
            hidden = np.maximum(hidden_pre, 0) # ReLU
            logits = hidden @ W2 + b2          # (B, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = x_flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            grad_x_flat = grad_hidden @ W1.T   # (B, d*E)
            grad_x_emb = grad_x_flat.reshape(B, d, E)
            # Efficient scatter-add using np.add.at
            np.add.at(grad_emb, xb, grad_x_emb)

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

    # Write weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray()
    out += header
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
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weight format (little-endian):
 *   int32 magic=0x454D4244, d=16, E=8, H=32, C=65
 *   float32 emb[256][E]
 *   float32 W1[d*E][H]   (16 blocks of E x H, row-major)
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    /* Expected weight size:
     * 5 int32 (20 bytes) + 256*E + d*E*H + H + H*C + C floats (4 bytes each)
     */
    long expected_floats = 256L * E + (long)d * E * H + H + (long)H * C + C;
    long expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* hidden = b1 */
        memcpy(hidden, b1, H * sizeof(float));

        /* Accumulate embeddings * W1 */
        for (int j = 0; j < d; j++) {
            const float *emb_row = emb + xi[j] * E;
            const float *W1_block = W1 + j * E * H;   // E x H
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e = 0; e < E; e++) {
                    sum += emb_row[e] * W1_block[e * H + h];
                }
                hidden[h] += sum;
            }
        }

        /* ReLU */
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        /* Output scores = b2 + hidden * W2 */
        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = b2[c];
            for (int h = 0; h < H; h++) {
                score += hidden[h] * W2[h * C + c];
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    free(hidden);
    return 0;
}
```