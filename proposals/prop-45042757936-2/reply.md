```hypothesis
Family: label-embed (embedding + 1-hidden-layer MLP). Each of the 16 context positions gets a learned byte embedding (dim E); embeddings are summed, passed through a ReLU layer (size H), then a linear output layer (65 classes). This captures pairwise character similarities and position-specific patterns with far fewer parameters than one-hot linear, enabling a hidden layer for non-linear interactions at low inference cost. If dev accuracy does not exceed label-linear (0.3375), the task is largely linear or E/H are too small.
```
```python
"""Embedding + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (E, H, lr, L2, epochs) chosen on last 20% of train.in (tuning split),
then final model trained on all train.in. Exports weights.bin:
magic 0x454D4244, d, E, H, C; float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
E_CHOICES = [32, 48, 64]
H_CHOICES = [64, 96, 128]
LR_CHOICES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES_PER_EPOCH = 600000  # virtual samples per epoch for early stopping

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def make_batches(n, batch, rng):
    order = rng.permutation(n)
    for s in range(0, n, batch):
        yield order[s:s + batch]

def train_model(X, y, C, E, H, lr, rng, epochs, batch, X_val=None, y_val=None):
    n, d = X.shape
    # Embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        for idx in make_batches(n, batch, rng):
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]
            # Forward
            # Sum embeddings: (B, E)
            h = np.zeros((B, E), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            # Hidden layer
            a = h @ W1 + b1
            h_relu = np.maximum(a, 0)
            # Output
            logits = h_relu @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_out = probs / B
            # Backprop
            grad_W2 = h_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h_relu = grad_out @ W2.T
            grad_a = grad_h_relu * (a > 0)
            grad_W1 = h.T @ grad_a + L2 * W1
            grad_b1 = grad_a.sum(axis=0)
            grad_h = grad_a @ W1.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation check
        if X_val is not None:
            val_acc = evaluate(X_val, y_val, C, params)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]
    if best_params is not None:
        params = best_params
    return params

def evaluate(X, y, C, params):
    emb, W1, b1, W2, b2 = params
    n, d = X.shape
    h = np.zeros((n, W1.shape[0]), np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    h = np.maximum(h @ W1 + b1, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n_all)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_score = -1.0
    best_hp = None
    best_params = None
    # Grid search
    for E in E_CHOICES:
        for H in H_CHOICES:
            for lr in LR_CHOICES:
                params = train_model(X_tr, y_tr, C, E, H, lr, rng, MAX_EPOCHS, BATCH, X_val, y_val)
                acc = evaluate(X_val, y_val, C, params)
                if acc > best_score:
                    best_score = acc
                    best_hp = (E, H, lr)
                    best_params = params
    print(f"Best hp: E={best_hp[0]}, H={best_hp[1]}, lr={best_hp[2]}, val_acc={best_score:.4f}", file=sys.stderr)
    # Retrain on all data with best hp (no early stopping, fixed epochs)
    E, H, lr = best_hp
    final_params = train_model(X_all, y_all, C, E, H, lr, rng, MAX_EPOCHS, BATCH)
    emb, W1, b1, W2, b2 = final_params
    # Pack weights
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
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
/* Embedding + 1-hidden-layer MLP for next-byte prediction.
 * ./infer weights.bin in.rec N
 * Weights format: magic 0x454D4244, d, E, H, C (int32); then float32 arrays:
 * emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * E);
    float *relu = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(hidden, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) hidden[e_idx] += e[e_idx];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e_idx = 0; e_idx < E; e_idx++) sum += hidden[e_idx] * W1[(long)e_idx * H + h];
            relu[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += relu[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```