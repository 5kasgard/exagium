```hypothesis
We propose label-embmlp: a shared token embedding (256×E) plus positional embeddings (16×E), concatenated and fed to a ReLU hidden layer (H) then a linear output (65). This captures character similarities and position–content interactions with far fewer parameters than one-hot linear (≈45k vs 266k), so it should generalize better on 24k examples. If dev accuracy ≤0.3375 or inference cost exceeds label-linear by >2×, the hypothesis fails.
```
```python
"""label-embmlp: shared token embeddings + positional embeddings + one ReLU layer.
train.py  card.json  train.in  train.key  weights.bin
Deterministic (seed 1), single-thread, numpy only."""
import json, math, struct, sys
import numpy as np

# Hyperparameters chosen on last 20% of train.in (tuning split)
E = 32          # token/position embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-3
BATCH = 128
MAX_EPOCHS = 50
PATIENCE = 5    # early stopping on tuning split

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
    C = card["classes"]
    d = card["input"]["bytes"]          # 16
    n = X_bytes.shape[0]

    # Split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)                    # shuffle once for representative split
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_bytes[train_idx], y[train_idx]
    X_val, y_val = X_bytes[val_idx], y[val_idx]
    n_tr = X_tr.shape[0]

    # Initialize parameters
    tok_emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_emb = (rng.uniform(-1, 1, (d, E)) * math.sqrt(6.0 / d)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [tok_emb, pos_emb, W1, b1, W2, b2]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, training=True):
        # X: (batch, d) uint8
        batch = X.shape[0]
        # Embedding lookup: (batch, d, E)
        tok = tok_emb[X]                 # (batch, d, E)
        pos = pos_emb[np.newaxis, :, :]  # (1, d, E)
        emb = tok + pos                  # (batch, d, E)
        flat = emb.reshape(batch, d * E) # (batch, d*E)
        h = flat @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        return logits, h_relu, flat

    def loss_grad(logits, y):
        # Softmax cross-entropy gradient
        max_log = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_log)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        probs /= len(y)
        return probs.astype(np.float32)

    best_val_acc = -1.0
    best_params = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx_batch = order[s:s + BATCH]
            Xb, yb = X_tr[idx_batch], y_tr[idx_batch]
            logits, h_relu, flat = forward(Xb)
            g = loss_grad(logits, yb)                     # (batch, C)

            # Gradients
            gW2 = h_relu.T @ g
            gb2 = g.sum(axis=0)
            gh_relu = g @ W2.T
            gh = gh_relu * (h_relu > 0)
            gW1 = flat.T @ gh
            gb1 = gh.sum(axis=0)
            gflat = gh @ W1.T                              # (batch, d*E)
            gemb = gflat.reshape(Xb.shape[0], d, E)        # (batch, d, E)
            # tok_emb grad: scatter-add over batch and positions
            gtok = np.zeros_like(tok_emb)
            gpos = np.zeros_like(pos_emb)
            for j in range(d):
                bytes_j = Xb[:, j]
                np.add.at(gtok, bytes_j, gemb[:, j, :])
                gpos[j] += gemb[:, j, :].sum(axis=0)

            grads = [gtok, gpos, gW1, gb1, gW2, gb2]
            # L2 regularization
            for i, p in enumerate(params):
                grads[i] += L2 * p

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # Retrain on full data with best params (or just use best_params)
    # We'll just use best_params and do a few more epochs on full data for stability
    params = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(3):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            Xb, yb = X_bytes[idx_batch], y[idx_batch]
            logits, h_relu, flat = forward(Xb)
            g = loss_grad(logits, yb)
            gW2 = h_relu.T @ g
            gb2 = g.sum(axis=0)
            gh_relu = g @ W2.T
            gh = gh_relu * (h_relu > 0)
            gW1 = flat.T @ gh
            gb1 = gh.sum(axis=0)
            gflat = gh @ W1.T
            gemb = gflat.reshape(Xb.shape[0], d, E)
            gtok = np.zeros_like(tok_emb)
            gpos = np.zeros_like(pos_emb)
            for j in range(d):
                bytes_j = Xb[:, j]
                np.add.at(gtok, bytes_j, gemb[:, j, :])
                gpos[j] += gemb[:, j, :].sum(axis=0)
            grads = [gtok, gpos, gW1, gb1, gW2, gb2]
            for i, p in enumerate(params):
                grads[i] += L2 * p
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    tok_emb, pos_emb, W1, b1, W2, b2 = params

    # Write weights.bin
    # Magic: 0x454D424C ("L B M E" in little-endian ASCII)
    # d, E, H, C as int32
    # Then floats in order: tok_emb (256*E), pos_emb (d*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    header = struct.pack("<5i", 0x454D424C, d, E, H, C)
    body = (
        tok_emb.astype("<f4").tobytes() +
        pos_emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d=16, E, H, C=65
 *   float tok_emb[256][E], pos_emb[d][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected weight size in bytes
    long expected = 20
        + 4L * (256 * E)          // tok_emb
        + 4L * (d * E)            // pos_emb
        + 4L * (d * E * H)        // W1
        + 4L * H                  // b1
        + 4L * (H * C)            // W2
        + 4L * C;                 // b2
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *tok_emb = (const float *)(hd + 5);
    const float *pos_emb = tok_emb + 256 * E;
    const float *W1 = pos_emb + d * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding: for each position j, emb[j*E:(j+1)*E] = tok_emb[xi[j]] + pos_emb[j]
        for (int j = 0; j < d; j++) {
            const float *tok = tok_emb + (long)xi[j] * E;
            const float *pos = pos_emb + (long)j * E;
            float *dst = emb + (long)j * E;
            for (int e = 0; e < E; e++) dst[e] = tok[e] + pos[e];
        }

        // Hidden layer: hidden = ReLU(emb @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h;  // W1 is (d*E) x H, column-major? No, row-major: W1[(d*E)*h + idx]
            for (int idx = 0; idx < d * E; idx++) sum += emb[idx] * w[idx * H];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = hidden @ W2 + b2
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