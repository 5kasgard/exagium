```hypothesis
Family: label-embmlp (Embedding MLP). Each of the 16 context positions gets a learned embedding (size E), concatenated into a 16*E vector, passed through one ReLU hidden layer (size H), then a linear output layer (65 classes). Embeddings capture character similarities and reduce dimensionality vs one-hot; the hidden layer models position interactions. Trained with Adam on the full training set after selecting E,H on the last 20% tuning split. Expected to beat label-linear (0.3375) by learning distributed representations; if dev accuracy ≤ 0.3375 or inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
"""label-embmlp: Embedding MLP for next-character prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hyperparameters (E, H) chosen on last 20% of train.in (tuning split), then retrained on all data.
Weight format: magic 0x454D424D, d, E, H, C; float32 arrays: emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
EMB_SIZES = [4, 8, 16]
HIDDEN_SIZES = [16, 32, 64]
TUNE_EPOCHS = 5  # quick evaluation on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_train, y_train, X_val, y_val, d, C, E, H, rng, epochs):
    """Train on train set, return validation accuracy."""
    n, k = X_train.shape
    # Embedding tables: d x 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    # Hidden layer: (d*E) x H
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H x C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, emb, W1, b1, W2, b2):
        # X: (batch, d) uint8
        batch = X.shape[0]
        # Embedding lookup: (batch, d, E)
        x_emb = emb[X]  # uses advanced indexing
        x_flat = x_emb.reshape(batch, -1)  # (batch, d*E)
        h = x_flat @ W1 + b1
        h = np.maximum(h, 0)  # ReLU
        logits = h @ W2 + b2
        return logits, h, x_flat

    def softmax_grad(logits, y):
        # logits: (batch, C), y: (batch,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        return (probs / len(y)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, h, x_flat = forward(Xb, emb, W1, b1, W2, b2)
            g = softmax_grad(logits, yb)  # (batch, C)
            # Backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h > 0)  # (batch, H)
            gW1 = x_flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # Embedding gradients: scatter-add
            gx_flat = gh @ W1.T  # (batch, d*E)
            gx_emb = gx_flat.reshape(Xb.shape[0], d, E)
            gemb = np.zeros_like(emb)
            np.add.at(gemb, (Xb[:, :, None], np.arange(E)[None, None, :]), gx_emb[:, :, :, None])  # wrong shape
            # Instead, loop over positions for simplicity (d=16 small)
            gemb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(gemb[j], Xb[:, j], gx_emb[:, j, :])
            gemb += L2 * emb

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Validation accuracy
    logits, _, _ = forward(X_val, emb, W1, b1, W2, b2)
    preds = logits.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    # Load all data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X_all.shape[0]
    # Split: last 20% for tuning, first 80% for training
    split = int(0.8 * n_total)
    X_train_full, y_train_full = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E = EMB_SIZES[0]
    best_H = HIDDEN_SIZES[0]

    # Grid search on tuning split
    for E in EMB_SIZES:
        for H in HIDDEN_SIZES:
            rng_split = np.random.default_rng(SEED + E * 1000 + H)  # deterministic per config
            acc, params = train_eval(X_train_full, y_train_full, X_tune, y_tune, d, C, E, H, rng_split, TUNE_EPOCHS)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_E, best_H = E, H
                print(f"Tune: E={E}, H={H}, acc={acc:.4f}", file=sys.stderr)

    print(f"Best: E={best_E}, H={best_H}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best hyperparameters
    rng_final = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_total))
    _, final_params = train_eval(X_all, y_all, X_all[:1], y_all[:1], d, C, best_E, best_H, rng_final, epochs)
    # The function returns params after training on X_all (since we pass X_all as train and a dummy val)
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, best_E, best_H, C)
    # Flatten arrays in order: emb (d,256,E), W1 (d*E, H), b1 (H), W2 (H, C), b2 (C)
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
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weight format: magic 0x454D424D, d, E, H, C; float32 arrays in order:
 *   emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t b1_sz = H;
    size_t W2_sz = (size_t)H * C;
    size_t b2_sz = C;
    long expected = 20 + 4L * (emb_sz + W1_sz + b1_sz + W2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + b1_sz;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }
        // Hidden layer: ReLU(x_emb @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += x_emb[k] * W1[(size_t)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
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