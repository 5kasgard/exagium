```hypothesis
Family: label-embed-mlp (shared byte embeddings + one ReLU hidden layer).  
Why: The 16-byte context has strong local dependencies; embeddings capture byte similarities and the hidden layer models interactions, while reducing the 4096 one-hot features to 16*E (E=32) for much faster inference than label-linear.  
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear, the family does not improve on the baseline.
```

```python
"""label-embed-mlp: shared byte embeddings + one ReLU hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters tuned on last 20% of train.in (file order), then retrained on all train.
Weights format (little-endian):
  magic=0x454D4244 ('EMBD'), d, E, H, C (int32)
  embed[256][E] (float32), W1[d*E][H], b1[H], W2[H][C], b2[C] (all float32, row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Grid for tuning (E, H)
GRID = [(16, 64), (16, 128), (32, 64), (32, 128), (32, 256), (64, 128)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, E, H, rng, n_samples, max_epochs):
    """Train and return (embed, W1, b1, W2, b2)."""
    n = X.shape[0]
    # Initialize parameters
    embed = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_idx):
        # X_idx: (n, d) uint8
        # Embedding lookup: (n, d, E)
        emb = embed[X_idx]  # (n, d, E)
        flat = emb.reshape(n, d * E)  # (n, d*E)
        hidden = flat @ W1 + b1
        hidden_relu = np.maximum(hidden, 0)
        logits = hidden_relu @ W2 + b2
        return flat, hidden, hidden_relu, logits

    def softmax_grad(logits, y_idx):
        # logits: (n, C), y_idx: (n,)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs / len(y_idx)

    for _ in range(min(max_epochs, math.ceil(n_samples / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            flat, hidden, hidden_relu, logits = forward(Xb)
            g = softmax_grad(logits, yb)  # (batch, C)

            # Backprop
            gW2 = hidden_relu.T @ g
            gb2 = g.sum(0)
            g_hidden_relu = g @ W2.T
            g_hidden = g_hidden_relu * (hidden > 0)
            gW1 = flat.T @ g_hidden
            gb1 = g_hidden.sum(0)
            g_flat = g_hidden @ W1.T  # (batch, d*E)
            # Embedding gradients: scatter-add
            g_embed = np.zeros_like(embed)
            g_flat_3d = g_flat.reshape(-1, d, E)
            Xb_flat = Xb.reshape(-1)
            # Efficient scatter-add using np.add.at
            for j in range(d):
                np.add.at(g_embed, Xb[:, j], g_flat_3d[:, j, :])

            grads = [g_embed + L2 * embed, gW1 + L2 * W1, gb1, gW2 + L2 * W2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return embed, W1, b1, W2, b2


def evaluate(X, y, d, C, E, H, embed, W1, b1, W2, b2):
    emb = embed[X]
    flat = emb.reshape(len(X), d * E)
    hidden = flat @ W1 + b1
    hidden_relu = np.maximum(hidden, 0)
    logits = hidden_relu @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split: last 20% for tuning
    n = len(X_all)
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None

    for E, H in GRID:
        print(f"Trying E={E}, H={H}", file=sys.stderr)
        params = train_model(X_train, y_train, d, C, E, H, rng, SAMPLES, MAX_EPOCHS)
        acc = evaluate(X_tune, y_tune, d, C, E, H, *params)
        print(f"  tune acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, (E, H), params

    E, H = best_cfg
    print(f"Best: E={E}, H={H}, acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data
    final_params = train_model(X_all, y_all, d, C, E, H, rng, SAMPLES, MAX_EPOCHS)
    embed, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten all arrays in row-major order (C-contiguous)
    out = header
    out += embed.astype("<f4").tobytes()
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
 * Input record: d bytes (d=16). Output: one byte class index (0..C-1).
 * Weights format (little-endian):
 *   magic=0x454D4244, d, E, H, C (int32)
 *   embed[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (float32, row-major). */
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
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + 256L * E;
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

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = embed + (long)xi[j] * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: flat (1 x dE) * W1 (dE x H) + b1
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += flat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f; // ReLU
        }

        // Output layer: hidden (1 x H) * W2 (H x C) + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```