```hypothesis
We propose "label-embmlp": position-specific byte embeddings (size K) concatenated into a 16*K vector, then a ReLU hidden layer (size H), then a linear output layer (65 classes). This preserves positional information while drastically reducing the input dimension compared to one-hot (16*K vs 4096). Tuning K and H on the last 20% of train.in should yield a model that beats label-linear (0.3375) in accuracy with similar or lower CPU cost (first layer 16*K*H multiply-adds vs label-linear's 16*256*H when H>0). Failure: dev accuracy ≤ label-linear or instructions/item significantly higher.
```
```python
"""label-embmlp: position-specific embeddings + 2-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Weights format: magic 0x454D424C, d=16, K, H, C; float32 embeddings[16][256][K],
W1[16*K][H], b1[H], W2[H][C], b2[C] (all row-major)."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning (K, H)
K_CHOICES = [8, 16, 32]
H_CHOICES = [32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, K, H, C, rng):
    # embeddings: (d, 256, K)
    emb = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (256 + K))).astype(np.float32)
    # W1: (d*K, H)
    W1 = (rng.uniform(-1, 1, (d * K, H)) * math.sqrt(6.0 / (d * K + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(emb, W1, b1, W2, b2, X_batch):
    """X_batch: (B, d) uint8. Returns logits (B, C), hidden (B, H), concat (B, d*K)."""
    B, d = X_batch.shape
    K = emb.shape[2]
    # Gather embeddings: for each position, select row
    concat = np.empty((B, d * K), dtype=np.float32)
    for j in range(d):
        concat[:, j*K:(j+1)*K] = emb[j, X_batch[:, j]]
    # Hidden layer
    hidden = concat @ W1 + b1
    np.maximum(hidden, 0, out=hidden)  # ReLU in-place
    # Output layer
    logits = hidden @ W2 + b2
    return logits, hidden, concat

def loss_and_grad(logits, y, hidden, concat, emb, W1, b1, W2, b2, L2):
    """Compute gradients for a batch. Returns (loss, grads_dict)."""
    B = logits.shape[0]
    # Softmax + cross-entropy gradient
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    probs /= B  # dL/dlogits
    loss = -np.log(np.maximum(probs[np.arange(B), y] + 1.0, 1e-12)).mean()  # not exactly, but we can compute properly
    # Actually compute loss correctly:
    log_probs = logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(B), y].mean()
    # L2 regularization (on weights only, not biases or embeddings? We'll regularize all weights)
    l2_loss = 0.5 * L2 * (np.sum(emb*emb) + np.sum(W1*W1) + np.sum(W2*W2))
    loss += l2_loss

    # Gradients
    # dL/dW2 = hidden.T @ probs
    grad_W2 = hidden.T @ probs + L2 * W2
    grad_b2 = probs.sum(axis=0)
    # dL/dhidden = probs @ W2.T
    grad_hidden = probs @ W2.T
    # ReLU backward
    grad_hidden[hidden == 0] = 0.0
    # dL/dW1 = concat.T @ grad_hidden
    grad_W1 = concat.T @ grad_hidden + L2 * W1
    grad_b1 = grad_hidden.sum(axis=0)
    # dL/dconcat = grad_hidden @ W1.T
    grad_concat = grad_hidden @ W1.T
    # dL/dembeddings: scatter add
    B, d = concat.shape[0], emb.shape[0]
    K = emb.shape[2]
    grad_emb = np.zeros_like(emb)
    for j in range(d):
        # grad_concat[:, j*K:(j+1)*K] corresponds to position j
        # We need to accumulate gradients for each byte value
        # Use np.add.at for scatter add
        np.add.at(grad_emb[j], X_batch[:, j], grad_concat[:, j*K:(j+1)*K])
    grad_emb += L2 * emb
    return loss, grad_emb, grad_W1, grad_b1, grad_W2, grad_b2

def evaluate(emb, W1, b1, W2, b2, X, y):
    logits, _, _ = forward(emb, W1, b1, W2, b2, X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_model(X_tr, y_tr, X_val, y_val, d, K, H, C, rng):
    emb, W1, b1, W2, b2 = init_params(d, K, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_val = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]
            logits, hidden, concat = forward(emb, W1, b1, W2, b2, X_batch)
            loss, g_emb, g_W1, g_b1, g_W2, g_b2 = loss_and_grad(
                logits, y_batch, hidden, concat, emb, W1, b1, W2, b2, L2)
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        val_acc = evaluate(emb, W1, b1, W2, b2, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_overall = -1.0
    best_hparams = None
    best_params = None

    for K in K_CHOICES:
        for H in H_CHOICES:
            val_acc, params = train_model(X_tr, y_tr, X_val, y_val, d, K, H, C, rng)
            print(f"K={K}, H={H}: val_acc={val_acc:.4f}", file=sys.stderr)
            if val_acc > best_overall:
                best_overall = val_acc
                best_hparams = (K, H)
                best_params = params

    print(f"Best: K={best_hparams[0]}, H={best_hparams[1]}, val_acc={best_overall:.4f}", file=sys.stderr)
    # Retrain on full data with best hyperparameters
    K, H = best_hparams
    rng_full = np.random.default_rng(SEED)  # fresh seed for determinism
    _, full_params = train_model(X, y, X_val, y_val, d, K, H, C, rng_full)  # we don't need val_acc, but function requires val set; we can pass dummy
    # Actually, we should train on full data without early stopping. Let's just run a fixed number of epochs on full data.
    # We'll reuse train_model but with X_tr=X, y_tr=y, and a dummy val set (we can use a small subset or just ignore val_acc).
    # To keep it simple, we'll train on full data for the same number of epochs as the best run.
    # But train_model uses early stopping based on val_acc. We'll instead do a final training on full data without validation.
    # We'll just run the same number of epochs as the best run had (which is epochs = min(MAX_EPOCHS, ceil(SAMPLES/n))).
    # Since n is larger now, epochs might be smaller. We'll just run for the same number of epochs as the best run used.
    # However, the best run used a different n (split). Let's compute epochs for full n.
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W1, b1, W2, b2 = init_params(d, K, H, C, rng_full)
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(epochs_full):
        order = rng_full.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X[idx]
            y_batch = y[idx]
            logits, hidden, concat = forward(emb, W1, b1, W2, b2, X_batch)
            loss, g_emb, g_W1, g_b1, g_W2, g_b2 = loss_and_grad(
                logits, y_batch, hidden, concat, emb, W1, b1, W2, b2, L2)
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424C  # "L B M E" in little-endian
    out = struct.pack("<5i", magic, d, K, H, C)
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
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, d, K, H, C (int32 each).
 * Then float32 arrays in order: emb[d][256][K], W1[d*K][H], b1[H], W2[H][C], b2[C] (row-major). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || K <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * K + (long)d * K * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * K;
    const float *b1 = W1 + (long)d * K * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = (float *)malloc(sizeof(float) * d * K);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Gather embeddings into concat
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * K;
            float *dst = concat + j * K;
            memcpy(dst, src, K * sizeof(float));
        }
        // Hidden layer: hidden = ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * K; idx++) {
                sum += concat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer: logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(concat); free(hidden); free(logits);
    return 0;
}
```