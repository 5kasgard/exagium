```hypothesis
We propose "label-embed-mlp": per-position byte embeddings (size E) concatenated, one ReLU hidden layer (size H), and a linear output layer. This captures position–character interactions more richly than a linear model while staying efficient with small E, H. We tune (E, H) on the last 20% of train.in, then retrain on all data. Expected dev accuracy > 0.34 (beating label-linear's 0.3375) at ~15k Ir/item. If accuracy ≤ 0.34, the added complexity is not justified.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU hidden layer.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Grid-searches (E, H) on tuning split (last 20% of train.in), then trains on all train.
Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d*256*E], W_hidden[d*E*H], b_hidden[H], W_out[H*C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Grid to search on tuning split
GRID = [(4, 16), (4, 32), (4, 64), (8, 16), (8, 32), (8, 64), (16, 16), (16, 32), (16, 64)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, rng, n_epochs):
    n = X.shape[0]
    # Parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_h = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_o = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b_o = np.zeros(C, np.float32)
    params = [emb, W_h, b_h, W_o, b_o]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            # Embeddings: (batch, d, E)
            emb_batch = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)  # (batch, d, E)
            flat = emb_batch.reshape(len(yb), d * E)  # (batch, d*E)
            h_pre = flat @ W_h + b_h
            h = np.maximum(h_pre, 0)
            logits = h @ W_o + b_o
            # Softmax + gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            g_out = (probs / len(yb)).astype(np.float32)  # (batch, C)
            # Backprop
            g_W_o = h.T @ g_out + L2 * W_o
            g_b_o = g_out.sum(0)
            g_h = g_out @ W_o.T
            g_h[h_pre <= 0] = 0
            g_W_h = flat.T @ g_h + L2 * W_h
            g_b_h = g_h.sum(0)
            g_flat = g_h @ W_h.T  # (batch, d*E)
            g_emb = np.zeros_like(emb)
            g_flat_reshaped = g_flat.reshape(len(yb), d, E)
            # Scatter add into g_emb
            for j in range(d):
                bytes_j = xb[:, j]
                np.add.at(g_emb[j], bytes_j, g_flat_reshaped[:, j])
            g_emb += L2 * emb
            grads = [g_emb, g_W_h, g_b_h, g_W_o, g_b_o]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def accuracy(X, y, d, C, E, H, params):
    emb, W_h, b_h, W_o, b_o = params
    n = X.shape[0]
    emb_batch = emb[np.arange(d)[:, None], X.T].transpose(1, 0, 2)
    flat = emb_batch.reshape(n, d * E)
    h = np.maximum(flat @ W_h + b_h, 0)
    logits = h @ W_o + b_o
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_EH = -1, None, None
    for E, H in GRID:
        # Deterministic seed per config
        config_seed = SEED * 10000 + E * 100 + H
        cfg_rng = np.random.default_rng(config_seed)
        params = train_model(X_train, y_train, d, C, E, H, cfg_rng, n_epochs)
        acc = accuracy(X_tune, y_tune, d, C, E, H, params)
        if acc > best_acc:
            best_acc, best_params, best_EH = acc, params, (E, H)

    # Retrain on full train with best config
    E, H = best_EH
    final_rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, d, C, E, H, final_rng,
                               min(MAX_EPOCHS, math.ceil(SAMPLES / n)))

    # Write weights.bin
    emb, W_h, b_h, W_o, b_o = final_params
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    header = struct.pack("<4i", magic, d, E, H, C)
    # All arrays float32 little-endian
    body = (emb.astype("<f4").tobytes() +
            W_h.astype("<f4").tobytes() +
            b_h.astype("<f4").tobytes() +
            W_o.astype("<f4").tobytes() +
            b_o.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d*256*E], W_hidden[d*E*H], b_hidden[H], W_out[H*C], b_out[C]. */
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
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W_h = emb + (long)d * 256 * E;
    const float *b_h = W_h + (long)d * E * H;
    const float *W_o = b_h + H;
    const float *b_o = W_o + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        // Hidden layer: h = ReLU(concat @ W_h + b_h)
        for (int u = 0; u < H; u++) {
            float sum = b_h[u];
            for (int v = 0; v < d * E; v++) sum += concat[v] * W_h[(long)v * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_o[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W_o[(long)u * C + c];
            scores[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```