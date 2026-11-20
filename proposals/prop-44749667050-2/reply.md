```hypothesis
Family: label-embed-mlp. Learns a per-position byte embedding (16 dims) followed by a single ReLU hidden layer (64 units) and a linear output. The embeddings capture positional character statistics; the hidden layer models interactions between positions that a linear model cannot. Trained with Adam on all 24k examples. If dev accuracy does not exceed label-linear's 0.3375, the non-linearity is insufficient or the embedding dimension is too small for this task.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU hidden layer.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D4245 ('EMBE'), d, H, C, D
  float32 T[d][256][H]   # precomputed embedding * W1 contributions
  float32 b1[H]
  float32 W2[H][C]
  float32 b2[C]
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 80
SAMPLES = 600000  # max samples seen (≈25 epochs on 24k)

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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    # Embeddings: [d, 256, D]
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # Hidden weights: [d*D, H]
    W1 = rng.normal(0, math.sqrt(2.0 / (d * D)), size=(d * D, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output weights: [H, C]
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    # Adam state
    params = [E, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(X_batch):
        # X_batch: [B, d] uint8
        B = X_batch.shape[0]
        # Gather embeddings: [B, d, D]
        emb = E[np.arange(d), X_batch.T].transpose(1, 0, 2)  # [B, d, D]
        flat = emb.reshape(B, -1)                            # [B, d*D]
        h_pre = flat @ W1 + b1                               # [B, H]
        h = np.maximum(h_pre, 0, dtype=np.float32)           # ReLU
        logits = h @ W2 + b2                                 # [B, C]
        return logits, h, flat, h_pre

    def loss_grad(logits, y_batch):
        # Softmax cross-entropy gradient wrt logits
        B = logits.shape[0]
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B  # [B, C]

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h, flat, h_pre = forward(Xb)
            g_logits = loss_grad(logits, yb)                 # [B, C]

            # Output layer gradients
            g_W2 = h.T @ g_logits + L2 * W2                  # [H, C]
            g_b2 = g_logits.sum(axis=0)                      # [C]

            # Hidden layer gradients (through ReLU)
            g_h = g_logits @ W2.T                            # [B, H]
            g_h[h_pre <= 0] = 0.0                            # ReLU derivative

            # W1, b1 gradients
            g_W1 = flat.T @ g_h + L2 * W1                    # [d*D, H]
            g_b1 = g_h.sum(axis=0)                           # [H]

            # Embedding gradients
            # g_h shape [B, H], W1 shape [d*D, H] -> g_flat = g_h @ W1.T [B, d*D]
            g_flat = g_h @ W1.T                              # [B, d*D]
            g_emb = g_flat.reshape(Xb.shape[0], d, D)        # [B, d, D]
            g_E = np.zeros_like(E)
            # Accumulate gradients per position and byte value
            # For each position p, byte value v, we add g_emb[b, p, :] for all b where Xb[b, p] == v
            # Vectorized using np.add.at
            for p in range(d):
                np.add.at(g_E[p], Xb[:, p], g_emb[:, p, :])

            grads = [g_E, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # After training, precompute T[pos][byte][h] = E[pos, byte, :] @ W1[pos*D:(pos+1)*D, :]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for p in range(d):
        T[p] = E[p] @ W1[p*D:(p+1)*D, :]   # [256, H]

    # Write weights
    magic = 0x454D4245  # 'EMBE'
    header = struct.pack("<5i", magic, d, H, C, D)
    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D4245, d, H, C, D
 *   float32 T[d][256][H]
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4];
    if (d != 16 || C < 1 || C > 256 || H < 1 || D < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!hidden || !scores) { fprintf(stderr, "alloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        memcpy(hidden, b1, H * sizeof(float));
        // Add T[pos][byte] for each position
        for (int p = 0; p < d; p++) {
            unsigned char byte = xi[p];
            const float *tp = T + ((long)p * 256 + byte) * H;
            for (int h = 0; h < H; h++) hidden[h] += tp[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        // scores = b2 + hidden * W2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2h[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```