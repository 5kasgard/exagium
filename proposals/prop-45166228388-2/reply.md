```hypothesis
Family: label-embmlp (embedding MLP). Learns a 32-d embedding per position/byte, sums them, then passes through a 64-unit ReLU layer to 65 logits. Non-linear combination of positions should capture n-gram interactions that linear models miss, while staying small and fast. If dev accuracy does not exceed label-linear's 0.3375, the added non-linearity is not helping on this data scale.
```
```python
"""Embedding MLP for next-character prediction: sum of per-position embeddings -> ReLU hidden -> output.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen after informal tuning on similar tasks)
E = 32          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
MAX_EPOCHS = 50
PATIENCE = 5    # early stopping on tuning split (last 20%)

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

    # Split: last 20% for validation (tuning split), first 80% for training
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(X_batch):
        # X_batch: (bs, d) uint8
        bs = X_batch.shape[0]
        # Sum embeddings: (bs, E)
        # For each position, gather embeddings and sum
        emb_sum = np.zeros((bs, E), np.float32)
        for j in range(d):
            emb_sum += emb[j, X_batch[:, j]]
        # Hidden layer
        hidden = emb_sum @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU in-place
        logits = hidden @ W2 + b2
        return emb_sum, hidden, logits

    def loss_and_grad(X_batch, y_batch):
        bs = X_batch.shape[0]
        emb_sum, hidden, logits = forward(X_batch)
        # Softmax cross-entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Gradient on logits
        grad_logits = probs
        grad_logits[np.arange(bs), y_batch] -= 1.0
        grad_logits /= bs
        # Gradients
        grad_W2 = hidden.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ W2.T
        grad_hidden[hidden == 0] = 0.0  # ReLU derivative
        grad_W1 = emb_sum.T @ grad_hidden + L2 * W1
        grad_b1 = grad_hidden.sum(axis=0)
        # Gradient on emb_sum
        grad_emb_sum = grad_hidden @ W1.T  # (bs, E)
        # Backprop to embeddings
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            # For each position, scatter-add grad_emb_sum to corresponding embedding rows
            idx = X_batch[:, j]
            np.add.at(grad_emb[j], idx, grad_emb_sum)
        grad_emb += L2 * emb
        return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2], probs

    best_val_acc = -1.0
    best_params = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            grads, _ = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation accuracy
        _, _, val_logits = forward(X_val)
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

    # Retrain on full data with best_params as initialization? 
    # Simpler: just use best_params (which were trained on 80%).
    # But contract says: choose settings on tuning split, then train on all of train.
    # We'll do a final short training on full data starting from best_params.
    if best_params is not None:
        params = best_params
        # Reset Adam state
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for epoch in range(5):  # few epochs on full data
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                X_batch, y_batch = X[idx], y[idx]
                grads, _ = loss_and_grad(X_batch, y_batch)
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1 - beta1) * gr
                    v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                    m_hat = m[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    emb, W1, b1, W2, b2 = params

    # Write weights.bin
    # Magic: 0x454D4C31 ("EML1")
    # int32: magic, d, E, H, C
    # float32: emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (all row-major)
    header = struct.pack("<5i", 0x454D4C31, d, E, H, C)
    out = bytearray(header)
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
/* Embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic(0x454D4C31), d, E, H, C (int32), then float32 arrays:
 *   emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (row-major). */
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
    if (nw < 20 || hd[0] != 0x454D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size in bytes
    long expected = 20 +
        (long)d * 256 * E * 4 +
        (long)E * H * 4 +
        (long)H * 4 +
        (long)H * C * 4 +
        (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb_sum = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!emb_sum || !hidden || !logits) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero emb_sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // Sum embeddings for each position
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // Hidden layer: hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
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