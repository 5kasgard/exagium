```hypothesis
We propose a "label-rnn" family: a small recurrent neural network with byte embeddings (emb=16, hidden=32) processing the 16-byte context sequentially. Since the task is next-character prediction, sequential modeling should capture local dependencies better than the bag-of-bytes linear model. The model is trained with Adam on cross-entropy. If dev accuracy does not exceed label-linear's 0.3375, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
SEED = 1
EMB_DIM = 16
HIDDEN_SIZE = 32
LR = 1e-3
BATCH = 64
EPOCHS = 30
L2 = 1e-5

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
    n, d = X_bytes.shape
    C = card["classes"]
    assert d == 16

    # Split last 20% for tuning (validation)
    split = int(0.8 * n)
    X_train, y_train = X_bytes[:split], y[:split]
    X_val, y_val = X_bytes[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: 256 x EMB_DIM
    scale = math.sqrt(3.0 / EMB_DIM)
    E = rng.uniform(-scale, scale, (256, EMB_DIM)).astype(np.float32)
    # W_xh: EMB_DIM x HIDDEN_SIZE
    scale = math.sqrt(6.0 / (EMB_DIM + HIDDEN_SIZE))
    W_xh = rng.uniform(-scale, scale, (EMB_DIM, HIDDEN_SIZE)).astype(np.float32)
    # W_hh: HIDDEN_SIZE x HIDDEN_SIZE
    scale = math.sqrt(6.0 / (HIDDEN_SIZE + HIDDEN_SIZE))
    W_hh = rng.uniform(-scale, scale, (HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32)
    b_h = np.zeros(HIDDEN_SIZE, np.float32)
    # W_hy: HIDDEN_SIZE x C
    scale = math.sqrt(6.0 / (HIDDEN_SIZE + C))
    W_hy = rng.uniform(-scale, scale, (HIDDEN_SIZE, C)).astype(np.float32)
    b_y = np.zeros(C, np.float32)

    params = [E, W_xh, W_hh, b_h, W_hy, b_y]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, 16) uint8
        B = X_batch.shape[0]
        h = np.zeros((B, HIDDEN_SIZE), np.float32)
        for step in range(16):
            x = X_batch[:, step]  # (B,)
            emb = E[x]            # (B, EMB_DIM)
            h = np.maximum(0, emb @ W_xh + h @ W_hh + b_h)  # ReLU
        logits = h @ W_hy + b_y   # (B, C)
        return logits, h

    def loss_and_grad(X_batch, y_batch):
        B = X_batch.shape[0]
        # Forward pass with intermediate storage for backward
        h_seq = []
        h = np.zeros((B, HIDDEN_SIZE), np.float32)
        for step in range(16):
            x = X_batch[:, step]
            emb = E[x]
            pre = emb @ W_xh + h @ W_hh + b_h
            h = np.maximum(0, pre)
            h_seq.append((emb, h, pre))
        logits = h @ W_hy + b_y
        # Softmax cross-entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()
        # Gradient on logits
        dlogits = probs
        dlogits[np.arange(B), y_batch] -= 1.0
        dlogits /= B
        # Gradients for output layer
        dW_hy = h.T @ dlogits + L2 * W_hy
        db_y = dlogits.sum(axis=0)
        dh = dlogits @ W_hy.T  # (B, HIDDEN_SIZE)
        # Backprop through time
        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        db_h = np.zeros_like(b_h)
        dE = np.zeros_like(E)
        for step in reversed(range(16)):
            emb, h_t, pre = h_seq[step]
            # ReLU gradient
            dh_pre = dh * (pre > 0)
            db_h += dh_pre.sum(axis=0)
            dW_xh += emb.T @ dh_pre
            dW_hh += (h_seq[step-1][1] if step > 0 else np.zeros((B, HIDDEN_SIZE))).T @ dh_pre
            dh = dh_pre @ W_hh.T
            dE_idx = X_batch[:, step]
            np.add.at(dE, dE_idx, dh_pre @ W_xh.T)
        return loss, [dE, dW_xh, dW_hh, db_h, dW_hy, db_y]

    # Training loop with validation-based early stopping (simple)
    best_val_acc = 0
    best_params = None
    for epoch in range(EPOCHS):
        order = rng.permutation(len(X_train))
        for s in range(0, len(X_train), BATCH):
            idx = order[s:s+BATCH]
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            loss, grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_logits, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on full dataset with best params? We'll just use the best params from validation.
    # But we should train on all data. We'll do a few more epochs on full data starting from best_params.
    params = best_params
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(5):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X_bytes[idx]
            y_batch = y[idx]
            loss, grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    E, W_xh, W_hh, b_h, W_hy, b_y = params

    # Write weights file
    # Magic: 0x524E4E01 ("RNN1" in little-endian)
    # Format: int32 magic, emb_dim, hidden_size, num_classes, input_len (16)
    # Then float32 arrays in order: E (256*emb_dim), W_xh (emb_dim*hidden), W_hh (hidden*hidden), b_h (hidden), W_hy (hidden*C), b_y (C)
    magic = 0x524E4E01
    header = struct.pack("<5i", magic, EMB_DIM, HIDDEN_SIZE, C, 16)
    arrays = [
        E.astype("<f4"),
        W_xh.astype("<f4"),
        W_hh.astype("<f4"),
        b_h.astype("<f4"),
        W_hy.astype("<f4"),
        b_y.astype("<f4"),
    ]
    out = header + b"".join(arr.tobytes() for arr in arrays)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x524E4E01) { fprintf(stderr, "bad weights\n"); return 3; }
    int emb_dim = hd[1], hidden = hd[2], C = hd[3], input_len = hd[4];
    if (input_len != 16) { fprintf(stderr, "unexpected input_len %d\n", input_len); return 3; }
    // Expected size: 20 + 4*(256*emb_dim + emb_dim*hidden + hidden*hidden + hidden + hidden*C + C)
    long expected = 20 + 4L * (256L*emb_dim + (long)emb_dim*hidden + (long)hidden*hidden + hidden + (long)hidden*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(wb + 20);
    const float *E = p; p += 256 * emb_dim;
    const float *W_xh = p; p += emb_dim * hidden;
    const float *W_hh = p; p += hidden * hidden;
    const float *b_h = p; p += hidden;
    const float *W_hy = p; p += hidden * C;
    const float *b_y = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * hidden);
    float *logits = malloc(sizeof(float) * C);
    float *emb = malloc(sizeof(float) * emb_dim);
    float *tmp = malloc(sizeof(float) * hidden);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != input_len) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, input_len); return 3; }

        // Initialize hidden state to zero
        memset(h, 0, sizeof(float) * hidden);

        for (int step = 0; step < input_len; step++) {
            // Embedding lookup
            const float *e = E + xi[step] * emb_dim;
            memcpy(emb, e, sizeof(float) * emb_dim);

            // Compute h = ReLU(emb @ W_xh + h @ W_hh + b_h)
            // tmp = emb @ W_xh
            for (int j = 0; j < hidden; j++) {
                float sum = 0.0f;
                for (int k = 0; k < emb_dim; k++) sum += emb[k] * W_xh[k * hidden + j];
                tmp[j] = sum;
            }
            // Add h @ W_hh
            for (int j = 0; j < hidden; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hidden; k++) sum += h[k] * W_hh[k * hidden + j];
                tmp[j] += sum;
            }
            // Add bias and ReLU
            for (int j = 0; j < hidden; j++) {
                float val = tmp[j] + b_h[j];
                h[j] = val > 0.0f ? val : 0.0f;
            }
        }

        // Output logits: h @ W_hy + b_y
        for (int c = 0; c < C; c++) {
            float sum = b_y[c];
            for (int j = 0; j < hidden; j++) sum += h[j] * W_hy[j * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(logits); free(emb); free(tmp);
    free(wb); if (x) free(x);
    return 0;
}
```