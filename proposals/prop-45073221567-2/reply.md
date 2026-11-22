```hypothesis
Family: label-embed-mlp. Uses per-position byte embeddings (E=8) followed by a ReLU hidden layer (H=64) and a linear output. Embeddings capture character similarities; the hidden layer models non-linear n-gram interactions. This should beat label-linear (0.3375) by learning distributed representations. If dev accuracy ≤ 0.34, the extra capacity does not help on this data size.
```
```python
"""label-embed-mlp: per-position embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses hyperparams on last 20% of train.in, then trains on all.
Weights format: magic 0x454D424C, int32 d,E,H,C; float32 table1[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
SEED = 1
E = 8          # embedding dimension per position
H = 64         # hidden layer size
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30
VAL_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split for validation (last 20% in file order)
    val_n = int(n * VAL_FRAC)
    train_n = n - val_n
    X_train, y_train = X[:train_n], y[:train_n]
    X_val, y_val = X[train_n:], y[train_n:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)   # (d, 256, E)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, E)
        x_emb = emb[np.arange(d), X_batch.T].transpose(1, 0, 2)  # (B, d, E)
        x_flat = x_emb.reshape(B, d * E)                         # (B, d*E)
        h_pre = x_flat @ W1 + b1                                 # (B, H)
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2                                     # (B, C)
        return logits, h_pre, h, x_flat

    def loss_grad(logits, y_batch):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B

    def evaluate(X_eval, y_eval):
        logits, _, _, _ = forward(X_eval)
        pred = logits.argmax(axis=1)
        return (pred == y_eval).mean()

    t = 0
    best_val = -1.0
    best_params = None

    for epoch in range(EPOCHS):
        order = rng.permutation(train_n)
        for s in range(0, train_n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, h_pre, h, x_flat = forward(Xb)
            g = loss_grad(logits, yb)                    # (B, C)

            # Gradients
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T                                # (B, H)
            gh[h_pre <= 0] = 0.0                         # ReLU grad
            gW1 = x_flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gx_flat = gh @ W1.T                          # (B, d*E)

            # Embedding gradients
            gemb = np.zeros_like(emb)
            gx_emb = gx_flat.reshape(Xb.shape[0], d, E)  # (B, d, E)
            for j in range(d):
                # Xb[:, j] are byte values for position j
                np.add.at(gemb[j], Xb[:, j], gx_emb[:, j, :])

            grads = [gemb, gW1, gb1, gW2, gb2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        val_acc = evaluate(X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on all data with best hyperparameters (using best_params as initialization)
    # We'll just continue training on full data for a few more epochs (or use best_params directly).
    # For simplicity, we use the best_params and do a few more epochs on full data.
    emb, W1, b1, W2, b2 = best_params
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for epoch in range(5):   # fine-tune on all data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h_pre, h, x_flat = forward(Xb)
            g = loss_grad(logits, yb)

            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[h_pre <= 0] = 0.0
            gW1 = x_flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gx_flat = gh @ W1.T

            gemb = np.zeros_like(emb)
            gx_emb = gx_flat.reshape(Xb.shape[0], d, E)
            for j in range(d):
                np.add.at(gemb[j], Xb[:, j], gx_emb[:, j, :])

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute table1: for each position j, each byte value v, the H-dim vector = emb[j,v] @ W1[j*E:(j+1)*E]
    table1 = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        table1[j] = emb[j] @ W1[j*E:(j+1)*E]   # (256, E) @ (E, H) -> (256, H)

    # Write weights
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<4i", magic, d, E, H, C)
    # table1[d][256][H], b1[H], W2[H][C], b2[C]
    out = header
    out += table1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: 1 byte class index.
 * Weights format: magic 0x454D424C, int32 d,E,H,C;
 *   float32 table1[d][256][H], b1[H], W2[H][C], b2[C].
 * Inference: acc = b1; for each position j: acc += table1[j][byte]; ReLU; scores = acc @ W2 + b2; argmax.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    (void)E; /* not used in inference */
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *table1 = (const float *)(hd + 5);
    const float *b1 = table1 + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *t = table1 + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```