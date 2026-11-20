```hypothesis
Family: label-embedding-mlp (shared byte embeddings + one ReLU hidden layer).
Why: Text has strong local patterns; shared embeddings capture byte similarities across positions with far fewer parameters than one-hot linear (256*E vs 16*256). A hidden layer adds non-linearity for context mixing. Expected to beat label-linear (0.3375) at similar or lower CPU cost.
Failure sign: Dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear (7359.9).
```

```python
"""label-embedding-mlp: shared byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D4244, d, E, H, C; float32 embed[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Hyperparameter grid (chosen on tuning split = last 20% of train.in)
GRID = [(4, 16), (4, 32), (4, 64), (8, 16), (8, 32), (8, 64), (16, 16), (16, 32), (16, 64)]
SEARCH_EPOCHS = 5


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_size = int(n * tune_frac)
    return X[:-tune_size], y[:-tune_size], X[-tune_size:], y[-tune_size:]


def init_params(E, H, C, d, rng):
    embed = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return embed, W1, b1, W2, b2


def forward(embed, W1, b1, W2, b2, X_idx):
    # X_idx: (batch, d) uint8
    batch, d = X_idx.shape
    E = embed.shape[1]
    # Embedding lookup
    emb = embed[X_idx.reshape(-1)].reshape(batch, d * E)  # (batch, d*E)
    # Hidden layer
    hidden = emb @ W1 + b1
    hidden = np.maximum(hidden, 0, dtype=np.float32)
    # Output
    logits = hidden @ W2 + b2
    return emb, hidden, logits


def loss_and_grad(embed, W1, b1, W2, b2, X_idx, y):
    batch, d = X_idx.shape
    E = embed.shape[1]
    H = W1.shape[1]
    C = W2.shape[1]

    # Forward
    emb, hidden, logits = forward(embed, W1, b1, W2, b2, X_idx)
    # Softmax cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    probs /= batch  # gradient of loss w.r.t logits

    # Backward
    # Output layer
    grad_W2 = hidden.T @ probs + L2 * W2
    grad_b2 = probs.sum(axis=0)
    # Hidden layer
    grad_hidden = probs @ W2.T
    grad_hidden *= (hidden > 0).astype(np.float32)
    grad_W1 = emb.T @ grad_hidden + L2 * W1
    grad_b1 = grad_hidden.sum(axis=0)
    # Embedding layer
    grad_emb_flat = grad_hidden @ W1.T  # (batch, d*E)
    grad_emb = np.zeros((256, E), np.float32)
    # Scatter-add gradients for each position
    for j in range(d):
        idx = X_idx[:, j]
        np.add.at(grad_emb, idx, grad_emb_flat[:, j * E:(j + 1) * E])
    grad_emb += L2 * embed

    loss = -np.log(probs[np.arange(batch), y] + 1e-12).mean()
    return loss, grad_emb, grad_W1, grad_b1, grad_W2, grad_b2


def train_loop(embed, W1, b1, W2, b2, X, y, epochs, rng):
    n = X.shape[0]
    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            loss, g_emb, g_W1, g_b1, g_W2, g_b2 = loss_and_grad(embed, W1, b1, W2, b2, Xb, yb)
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(embed, W1, b1, W2, b2, X, y):
    _, _, logits = forward(embed, W1, b1, W2, b2, X)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    rng = np.random.default_rng(SEED)

    # Split tuning set (last 20% in file order)
    X_tr, y_tr, X_tune, y_tune = split_train_tune(X, y, 0.2)

    # Hyperparameter search
    best_acc, best_cfg, best_params = -1, None, None
    for E, H in GRID:
        embed, W1, b1, W2, b2 = init_params(E, H, C, d, rng)
        embed, W1, b1, W2, b2 = train_loop(embed, W1, b1, W2, b2, X_tr, y_tr, SEARCH_EPOCHS, rng)
        acc = evaluate(embed, W1, b1, W2, b2, X_tune, y_tune)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, (E, H), (embed, W1, b1, W2, b2)

    # Retrain on full data with best config
    E, H = best_cfg
    embed, W1, b1, W2, b2 = init_params(E, H, C, d, rng)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))
    embed, W1, b1, W2, b2 = train_loop(embed, W1, b1, W2, b2, X, y, epochs, rng)

    # Export weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic(0x454D4244), d, E, H, C (int32); then float32 arrays:
 * embed[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header values\n"); return 3; }

    size_t embed_sz = 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t b1_sz = H;
    size_t W2_sz = (size_t)H * C;
    size_t b2_sz = C;
    size_t total_floats = embed_sz + W1_sz + b1_sz + W2_sz + b2_sz;
    if (nw != 20 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *embed = p; p += embed_sz;
    const float *W1 = p; p += W1_sz;
    const float *b1 = p; p += b1_sz;
    const float *W2 = p; p += W2_sz;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_vec = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = embed + xi[j] * E;
            memcpy(emb_vec + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(emb_vec * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_vec[k] * W1[k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
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