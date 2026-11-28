```hypothesis
Family: label-embed (positional byte embeddings + one ReLU hidden layer).
Why: The task predicts the next character from 16 context bytes. One-hot linear models (label-linear) ignore similarity between byte values (e.g., 'a' and 'b' are close). Learned embeddings per position capture such structure with far fewer parameters, and a small hidden layer adds nonlinear interaction. This should improve accuracy over label-linear (0.3375) while keeping inference fast (lookup + two matvecs).
Failure: If dev accuracy ≤ 0.3375 or instructions/item exceed label-linear by >2×, the family does not meet its goal.
```

```python
"""label-embed: positional byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  uint32 magic=0x454D4244 ('EMBD'), int32 d, H, C, d_in(=16)
  float32 emb[d_in][256][d]
  float32 W1[d_in*d][H], b1[H]
  float32 W2[H][C], b2[C]
"""
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D_EMB = 8          # embedding dimension per position
H = 32             # hidden layer size
SEED = 1
LR = 0.01
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

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int64
    # returns gradient w.r.t. logits: (B, C)
    max_logit = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d_in = card["input"]["bytes"]  # 16
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d_in, 256, D_EMB)
    emb = (rng.normal(0, 1, (d_in, 256, D_EMB)) * math.sqrt(1.0 / 256)).astype(np.float32)
    # W1: (d_in*D_EMB, H), b1: (H,)
    W1 = (rng.normal(0, 1, (d_in * D_EMB, H)) * math.sqrt(2.0 / (d_in * D_EMB))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C), b2: (C,)
    W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d_in)
            yb = y[idx]                # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d_in, D_EMB)
            emb_vecs = emb[np.arange(d_in)[None, :], xb]   # (B, d_in, D_EMB)
            flat = emb_vecs.reshape(B, -1)                 # (B, d_in*D_EMB)
            hidden_pre = flat @ W1 + b1                    # (B, H)
            hidden = np.maximum(hidden_pre, 0)             # ReLU
            logits = hidden @ W2 + b2                      # (B, C)

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)  # (B, C)

            # Backprop
            g_W2 = hidden.T @ g_logits + L2 * W2           # (H, C)
            g_b2 = g_logits.sum(axis=0)                    # (C,)
            g_hidden = g_logits @ W2.T                     # (B, H)
            g_hidden[hidden_pre <= 0] = 0                  # ReLU grad
            g_W1 = flat.T @ g_hidden + L2 * W1             # (d_in*D_EMB, H)
            g_b1 = g_hidden.sum(axis=0)                    # (H,)
            g_flat = g_hidden @ W1.T                       # (B, d_in*D_EMB)
            g_emb_vecs = g_flat.reshape(B, d_in, D_EMB)    # (B, d_in, D_EMB)

            # Gradient for embeddings: scatter-add
            g_emb = np.zeros_like(emb)
            np.add.at(g_emb, (np.arange(d_in)[None, :], xb), g_emb_vecs)
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<5i", magic, D_EMB, H, C, d_in)
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (context). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   uint32 magic=0x454D4244, int32 d_emb, H, C, d_in
 *   float32 emb[d_in][256][d_emb]
 *   float32 W1[d_in*d_emb][H], b1[H]
 *   float32 W2[H][C], b2[C]
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
    int d_emb = hd[1], H = hd[2], C = hd[3], d_in = hd[4];
    if (d_in != 16 || C != 65 || d_emb <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected weight size
    long expected = 20
        + (long)d_in * 256 * d_emb * 4
        + (long)d_in * d_emb * H * 4 + (long)H * 4
        + (long)H * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d_in * 256 * d_emb;
    const float *b1 = W1 + (long)d_in * d_emb * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_in) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_in); return 3; }

        // Embedding lookup and flatten
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int p = 0; p < d_in; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * d_emb;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int d = 0; d < d_emb; d++) sum += e[d] * W1[((long)p * d_emb + d) * H + h];
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            for (int c = 0; c < C; c++) logits[c] += hv * W2[(long)h * C + c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```