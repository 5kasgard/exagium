```hypothesis
Family: label-embedding. Learned per-position byte embeddings (dim E) summed, then one ReLU hidden layer (size H), then linear to 65 classes. Embeddings capture character similarities; hidden layer models interactions. Should beat label-linear (0.3375) by learning distributed representations. Failure: if dev accuracy ≤ 0.3375 or inference cost exceeds label-linear significantly.
```

```python
"""label-embedding: per-position byte embeddings summed, one ReLU hidden layer, linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (could be tuned on the last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 20
SAMPLES = 600000  # max samples per epoch
EMBED_DIM = 32
HIDDEN = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_crossentropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits (n, C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

def train_model(X, y, C, embed_dim, hidden, rng, epochs, batch, lr, l2):
    n, d = X.shape
    # Initialize embeddings: (d, 256, embed_dim)
    scale = math.sqrt(6.0 / (256 + embed_dim))
    embed = rng.uniform(-scale, scale, (d, 256, embed_dim)).astype(np.float32)
    # Hidden layer: embed_dim -> hidden
    scale = math.sqrt(6.0 / (embed_dim + hidden))
    W1 = rng.uniform(-scale, scale, (embed_dim, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    # Output layer: hidden -> C
    scale = math.sqrt(6.0 / (hidden + C))
    W2 = rng.uniform(-scale, scale, (hidden, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)

            # Embedding lookup and sum
            emb_sum = np.zeros((len(idx), embed_dim), np.float32)
            for j in range(d):
                emb_sum += embed[j, xb[:, j]]   # (batch, embed_dim)

            # Hidden layer
            h_pre = emb_sum @ W1 + b1
            h = np.maximum(h_pre, 0)

            # Output layer
            logits = h @ W2 + b2

            # Gradient
            g_logits = softmax_crossentropy_grad(logits, yb).astype(np.float32)   # (batch, C)

            # Output layer grads
            g_W2 = h.T @ g_logits + l2 * W2
            g_b2 = g_logits.sum(0)

            # Hidden layer grads
            g_h = g_logits @ W2.T
            g_h_pre = g_h * (h_pre > 0)

            g_W1 = emb_sum.T @ g_h_pre + l2 * W1
            g_b1 = g_h_pre.sum(0)

            # Embedding grads
            g_emb = np.zeros_like(embed)
            for j in range(d):
                # g_h_pre shape (batch, embed_dim), scatter-add to embed[j, xb[:, j]]
                np.add.at(g_emb[j], xb[:, j], g_h_pre)

            g_emb += l2 * embed

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params

def eval_model(X, y, C, params):
    embed, W1, b1, W2, b2 = params
    n, d = X.shape
    logits = np.zeros((n, C), np.float32)
    for start in range(0, n, 1024):
        end = min(start + 1024, n)
        xb = X[start:end]
        emb_sum = np.zeros((end-start, embed.shape[2]), np.float32)
        for j in range(d):
            emb_sum += embed[j, xb[:, j]]
        h = np.maximum(emb_sum @ W1 + b1, 0)
        logits[start:end] = h @ W2 + b2
    return accuracy(logits, y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Grid search over (embed_dim, hidden) on tuning split
    # We'll just use the fixed values defined at top; but we could search.
    # For simplicity, we use the fixed values.
    embed_dim, hidden = EMBED_DIM, HIDDEN

    # Determine epochs from sample budget
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    print(f"Training on {len(X_train)} samples, tuning on {len(X_tune)}, embed={embed_dim}, hidden={hidden}, epochs={epochs}", file=sys.stderr)

    params = train_model(X_train, y_train, C, embed_dim, hidden, rng, epochs, BATCH, LR, L2)

    tune_acc = eval_model(X_tune, y_tune, C, params)
    print(f"Tuning accuracy: {tune_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(X, y, C, embed_dim, hidden, rng, epochs_full, BATCH, LR, L2)

    # Pack weights.bin
    embed, W1, b1, W2, b2 = params
    magic = 0x454D4244  # 'EMBD' in little-endian
    header = struct.pack("<5i", magic, d, embed_dim, hidden, C)
    # Embeddings: d * 256 * embed_dim floats
    emb_bytes = embed.astype("<f4").tobytes()
    # Hidden layer
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    # Output layer
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights format: magic(0x454D4244), d, E, H, C (int32 each),
 *   embed[d][256][E] (float32), W1[E][H], b1[H], W2[H][C], b2[C]. */
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
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *row = embed + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += row[e];
        }

        // Hidden layer: ReLU(emb_sum * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
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
    return 0;
}
```