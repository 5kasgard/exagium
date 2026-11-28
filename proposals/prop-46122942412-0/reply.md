```hypothesis
Family: label-emb-mlp. Learns a small embedding per feature value (E=8) and a one-hidden-layer MLP (H=32). Embeddings capture similarities between categories and reduce dimensionality vs one-hot, allowing the MLP to model interactions efficiently. On Adult, this should exceed label-linear's 0.852 accuracy at similar or lower inference cost (embedding lookup + small matmul). If dev accuracy ≤0.852 or Ir/item ≥1000, the approach fails.
```

```python
"""label-emb-mlp: embedding + one-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"E": 4, "H": 16, "lr": 1e-3, "epochs": 30},
    {"E": 8, "H": 32, "lr": 3e-3, "epochs": 30},
    {"E": 8, "H": 64, "lr": 1e-3, "epochs": 30},
    {"E": 16, "H": 32, "lr": 3e-3, "epochs": 30},
]
SEED = 1
BATCH = 64
L2 = 1e-4
FINAL_EPOCHS = 50  # for full retrain


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, config, rng, epochs=None):
    """Train on given data, return (params, loss_history)."""
    n, d = X.shape
    C = 2
    E = config["E"]
    H = config["H"]
    lr = config["lr"]
    if epochs is None:
        epochs = config["epochs"]

    # Embeddings: (d, 256, E) -> flatten to (d*256, E)
    emb = (rng.uniform(-1, 1, (d * 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # MLP: input dim = d*E
    in_dim = d * E
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (bs, d) uint8
        bs = X_batch.shape[0]
        # Gather embeddings: (bs, d, E)
        emb_batch = emb[X_batch.reshape(-1)].reshape(bs, d, E)
        # Flatten to (bs, d*E)
        x = emb_batch.reshape(bs, in_dim)
        # Hidden
        h = x @ W1 + b1
        h = np.maximum(h, 0)
        # Output
        logits = h @ W2 + b2
        return logits, h, x

    def softmax_grad(logits, y_batch):
        # logits: (bs, C), y_batch: (bs,)
        max_log = logits.max(axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        probs = exp_log / exp_log.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_batch)), y_batch] -= 1.0
        return probs / len(y_batch)  # (bs, C)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, x = forward(xb)
            g_out = softmax_grad(logits, yb)  # (bs, C)

            # Gradients
            g_W2 = h.T @ g_out + L2 * W2
            g_b2 = g_out.sum(axis=0)
            g_h = g_out @ W2.T * (h > 0)
            g_W1 = x.T @ g_h + L2 * W1
            g_b1 = g_h.sum(axis=0)
            # Embedding gradients: scatter-add
            g_emb = np.zeros_like(emb)
            # xb shape (bs, d), g_h shape (bs, H), W1 shape (in_dim, H)
            # For each feature j, each sample i, each embedding dim k:
            # g_emb[j*256 + xb[i,j], k] += g_h[i, :] @ W1[j*E + k, :]
            # Vectorized:
            for j in range(d):
                vals = xb[:, j]  # (bs,)
                # W1 slice for feature j: (E, H)
                W1_j = W1[j * E:(j + 1) * E]
                # g_h @ W1_j.T -> (bs, E)
                grad_emb_j = g_h @ W1_j.T  # (bs, E)
                np.add.at(g_emb, j * 256 + vals, grad_emb_j)
            g_emb += L2 * emb

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


def evaluate(X, y, params):
    d = X.shape[1]
    E = params[0].shape[1]
    H = params[2].shape[0]
    in_dim = d * E
    emb, W1, b1, W2, b2 = params
    bs = X.shape[0]
    emb_batch = emb[X.reshape(-1)].reshape(bs, d, E)
    x = emb_batch.reshape(bs, in_dim)
    h = np.maximum(x @ W1 + b1, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_config = None
    best_params = None
    rng_base = np.random.default_rng(SEED)

    for idx, config in enumerate(GRID):
        rng = np.random.default_rng(SEED + idx + 1)
        params = train_model(X_train, y_train, config, rng)
        acc = evaluate(X_tune, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_params = params

    # Retrain on full data with best config
    rng_final = np.random.default_rng(SEED)
    final_params = train_model(X, y, best_config, rng_final, epochs=FINAL_EPOCHS)

    # Pack weights
    emb, W1, b1, W2, b2 = final_params
    E = emb.shape[1]
    H = W1.shape[1]
    magic = 0x454D424D  # "MBME" in little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E <= 0 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long in_dim = (long)d * E;
    long emb_size = (long)d * 256 * E;
    long w1_size = in_dim * H;
    long w2_size = (long)H * C;
    long expected = 20 + 4L * (emb_size + w1_size + H + w2_size + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_size;
    const float *b1 = W1 + w1_size;
    const float *W2 = b1 + H;
    const float *b2 = W2 + w2_size;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        for (int h = 0; h < H; h++) hidden[h] = b1[h];

        // Accumulate embeddings * W1
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int kk = 0; kk < E; kk++) {
                float ev = e[kk];
                if (ev == 0.0f) continue;
                const float *w1_row = W1 + ((long)j * E + kk) * H;
                for (int h = 0; h < H; h++) hidden[h] += ev * w1_row[h];
            }
        }

        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output scores = b2 + hidden * W2
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w2_row = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2_row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```