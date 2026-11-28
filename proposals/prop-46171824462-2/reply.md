```hypothesis
Family: label-emb-mlp (learned byte embeddings per position + one ReLU hidden layer).
Why: The linear model (label-linear) uses one-hot vectors per position, ignoring character similarities. Embedding each byte at each position into a dense vector (dim=32) captures semantic similarity between characters (e.g., 'a' and 'b' close) and allows a hidden layer (dim=64) to model position interactions non-linearly. This adds only ~2× the parameters of label-linear but should better model the sequential nature of text.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) it shows the task does not benefit from learned embeddings + shallow non-linearity, or the chosen dimensions are too small.
```

```python
"""label-emb-mlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D4231, d, E, H, C; then embeddings[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
SEED = 1
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 0.01
L2 = 1e-5
BATCH = 128
EPOCHS = 20

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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    E = EMBED_DIM
    H = HIDDEN_DIM
    scale = math.sqrt(6.0 / (256 + E))
    embeddings = rng.uniform(-scale, scale, (d, 256, E)).astype(np.float32)

    # Hidden layer: (E, H)
    scale = math.sqrt(6.0 / (E + H))
    W1 = rng.uniform(-scale, scale, (E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)

    # Output layer: (H, C)
    scale = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale, scale, (H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    # Adam state
    params = [embeddings, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Precompute indices for fast embedding lookup: for each position, we will index embeddings[pos, X[:, pos]]
    pos_idx = np.arange(d, dtype=np.int32)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X_bytes[idx]          # (B, d)
            yb = y[idx]                # (B,)

            # Embedding sum: (B, E)
            # embeddings[pos_idx, Xb.T] -> (d, B, E) -> transpose to (B, d, E) -> sum over d
            emb = embeddings[pos_idx, Xb.T].transpose(1, 0, 2)  # (B, d, E)
            emb_sum = emb.sum(axis=1)                           # (B, E)

            # Hidden layer with ReLU
            hidden_pre = emb_sum @ W1 + b1                      # (B, H)
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)

            # Output logits
            logits = hidden @ W2 + b2                           # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)  # (B, C)

            # Backprop output layer
            grad_W2 = hidden.T @ grad_logits + L2 * W2          # (H, C)
            grad_b2 = grad_logits.sum(axis=0)                   # (C,)

            # Backprop hidden layer
            grad_hidden = grad_logits @ W2.T                    # (B, H)
            grad_hidden[hidden_pre <= 0] = 0.0                  # ReLU gradient

            # Backprop hidden weights
            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1         # (E, H)
            grad_b1 = grad_hidden.sum(axis=0)                   # (H,)

            # Backprop embeddings
            grad_emb_sum = grad_hidden @ W1.T                   # (B, E)
            # Need to add grad_emb_sum to each position's embedding for the corresponding byte
            grad_embeddings = np.zeros_like(embeddings)         # (d, 256, E)
            # For each position p, for each example i, byte b = Xb[i, p], add grad_emb_sum[i] to grad_embeddings[p, b]
            # Vectorized: use np.add.at
            for p in range(d):
                np.add.at(grad_embeddings[p], Xb[:, p], grad_emb_sum)

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    magic = 0x454D4231  # "EMB1"
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
/* label-emb-mlp inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding sum
        memset(emb_sum, 0, sizeof(float) * E);
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * E;
            for (int j = 0; j < E; j++) emb_sum[j] += e[j];
        }

        // Hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h * E;  // W1 is (E, H) stored row-major? In train.py W1 shape (E, H), so W1[:, h] is column h.
            // Actually in train.py: W1 shape (E, H), so W1[j, h] is at index j*H + h. So column h is stride H.
            for (int j = 0; j < E; j++) sum += emb_sum[j] * w[j * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
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

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```