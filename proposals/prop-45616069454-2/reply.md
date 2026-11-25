```hypothesis
Family: label-embed. Learned per-position byte embeddings (size E) concatenated and fed to a single ReLU hidden layer (size H) then linear output. Captures positional interactions more efficiently than one-hot linear models, should exceed label-linear's 0.3375 on this character-prediction task. If dev accuracy ≤ 0.3375, the approach fails.
```

```python
"""label-embed: per-position embeddings + one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on the tuning split (last 20% of train)
GRID = [
    {"E": 32, "H": 64},
    {"E": 64, "H": 128},
    {"E": 32, "H": 128},
]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]

def accuracy(logits, y):
    return (logits.argmax(1) == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split: last 20% for tuning
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, 0.2)
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)
    best_cfg = None
    best_val = -1.0
    best_params = None

    for cfg in GRID:
        E, H = cfg["E"], cfg["H"]
        W_emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256))).astype(np.float32)
        W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W_emb, W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

        max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        t = 0
        for epoch in range(max_epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                # Embedding lookup and concat
                emb = W_emb[np.arange(d)[:, None], xb.T]  # (d, batch, E)
                emb = emb.transpose(1, 0, 2).reshape(len(yb), d * E)
                # Forward
                h = emb @ W1 + b1
                h_relu = np.maximum(h, 0)
                logits = h_relu @ W2 + b2
                # Softmax + cross-entropy gradient
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(len(yb)), yb] -= 1.0
                grad_out = (probs / len(yb)).astype(np.float32)
                # Backprop
                grad_W2 = h_relu.T @ grad_out + L2 * W2
                grad_b2 = grad_out.sum(0)
                grad_h_relu = grad_out @ W2.T
                grad_h = grad_h_relu * (h > 0)
                grad_W1 = emb.T @ grad_h + L2 * W1
                grad_b1 = grad_h.sum(0)
                grad_emb_concat = grad_h @ W1.T
                grad_emb = grad_emb_concat.reshape(len(yb), d, E).transpose(1, 0, 2)
                grad_W_emb = np.zeros_like(W_emb)
                np.add.at(grad_W_emb, (np.arange(d)[:, None], xb.T), grad_emb)
                grad_W_emb += L2 * W_emb
                grads = [grad_W_emb, grad_W1, grad_b1, grad_W2, grad_b2]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        emb_val = W_emb[np.arange(d)[:, None], X_val.T].transpose(1, 0, 2).reshape(len(y_val), d * E)
        h_val = np.maximum(emb_val @ W1 + b1, 0)
        logits_val = h_val @ W2 + b2
        val_acc = accuracy(logits_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_cfg = cfg
            best_params = [p.copy() for p in params]

    # Retrain on full training set with best config
    E, H = best_cfg["E"], best_cfg["H"]
    W_emb, W1, b1, W2, b2 = best_params
    # Re-initialize with same seed for reproducibility? We'll continue from the best_params but that's not retraining from scratch.
    # The contract says "choose them on the tuning split only, then train on all of train".
    # We'll re-initialize and train on full train for the same number of epochs.
    rng = np.random.default_rng(SEED)
    W_emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n_full = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    t = 0
    for epoch in range(max_epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            emb = W_emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2).reshape(len(yb), d * E)
            h = emb @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_out = (probs / len(yb)).astype(np.float32)
            grad_W2 = h_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(0)
            grad_h_relu = grad_out @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            grad_emb_concat = grad_h @ W1.T
            grad_emb = grad_emb_concat.reshape(len(yb), d, E).transpose(1, 0, 2)
            grad_W_emb = np.zeros_like(W_emb)
            np.add.at(grad_W_emb, (np.arange(d)[:, None], xb.T), grad_emb)
            grad_W_emb += L2 * W_emb
            grads = [grad_W_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C, then embeddings (d*256*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D4245  # 'EMBE'
    W_emb_f = params[0].astype("<f4")
    W1_f = params[1].astype("<f4")
    b1_f = params[2].astype("<f4")
    W2_f = params[3].astype("<f4")
    b2_f = params[4].astype("<f4")
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header + W_emb_f.tobytes() + W1_f.tobytes() + b1_f.tobytes() + W2_f.tobytes() + b2_f.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D4245), d, E, H, C (int32),
 *   embeddings[d][256][E] (float32), W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    float *concat = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h; // column-major? W1 is (d*E) x H stored row-major? In train we did W1[d*E, H] and emb @ W1.
            // In train: W1 shape (d*E, H), so row-major: W1[i*H + h] for i-th input, h-th output.
            for (int idx = 0; idx < d * E; idx++) {
                sum += concat[idx] * W1[idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
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