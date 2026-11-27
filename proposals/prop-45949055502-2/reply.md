```hypothesis
Family: label-rnn (Elman RNN with shared byte embeddings). The 16-byte context is a short sequence; an RNN captures positional and sequential dependencies better than a linear model. With hidden size 64 (~16k parameters) it should exceed label-linear's 33.75% dev accuracy while staying under 15k Ir/item. If dev accuracy ≤33.75% or inference cost >15k Ir/item, the hypothesis fails.
```

```python
"""label-rnn: Elman RNN with shared embeddings for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Architecture: embedding(256→H), tanh recurrent (H→H), linear (H→C).
Trains with Adam (lr=1e-3, L2=1e-4) on 80% of train, validates on last 20%,
then retrains on full train for the best epoch count.
Weights format: magic 0x524E4E31, int32 d,H,C; float32 E[256*H], W_hh[H*H], b_h[H], W_out[H*C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
HIDDEN = 64  # will be validated on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (B, C)."""
    B = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B


def rnn_forward(params, X):
    """X: (B, 16) uint8. Returns (logits, hidden_states) where hidden_states[t] = h_t (B, H)."""
    E, W_hh, b_h, W_out, b_out = params
    B = X.shape[0]
    H = W_hh.shape[0]
    h = np.zeros((B, H), dtype=np.float32)
    hiddens = np.zeros((B, 16, H), dtype=np.float32)
    for t in range(16):
        e = E[X[:, t]]                     # (B, H)
        h = np.tanh(e + h @ W_hh.T + b_h)  # (B, H)
        hiddens[:, t] = h
    logits = h @ W_out.T + b_out
    return logits, hiddens


def rnn_backward(params, X, y, hiddens, dlogits):
    """Accumulate gradients into grads (list same structure as params)."""
    E, W_hh, b_h, W_out, b_out = params
    B = X.shape[0]
    H = W_hh.shape[0]
    dE = np.zeros_like(E)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    dW_out = np.zeros_like(W_out)
    db_out = np.zeros_like(b_out)

    # Output layer gradients
    dW_out = dlogits.T @ hiddens[:, -1]
    db_out = dlogits.sum(axis=0)
    dh = dlogits @ W_out  # (B, H)

    # Backprop through time
    for t in reversed(range(16)):
        h_t = hiddens[:, t]
        da = dh * (1.0 - h_t * h_t)  # derivative of tanh
        db_h += da.sum(axis=0)
        # embedding gradient
        np.add.at(dE, X[:, t], da)
        # recurrent gradient
        if t == 0:
            h_prev = np.zeros((B, H), dtype=np.float32)
        else:
            h_prev = hiddens[:, t - 1]
        dW_hh += da.T @ h_prev
        dh = da @ W_hh

    # L2 regularization gradients
    dE += L2 * E
    dW_hh += L2 * W_hh
    dW_out += L2 * W_out
    return [dE, dW_hh, db_h, dW_out, db_out]


def evaluate(params, X, y):
    logits, _ = rnn_forward(params, X)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16

    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    N = X.shape[0]

    # Split 80/20 for tuning
    split = int(0.8 * N)
    idx = np.arange(N)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Initialize parameters
    H = HIDDEN
    scale_emb = math.sqrt(3.0 / 256)
    scale_rec = math.sqrt(3.0 / H)
    scale_out = math.sqrt(3.0 / H)
    E = (rng.uniform(-scale_emb, scale_emb, (256, H))).astype(np.float32)
    W_hh = (rng.uniform(-scale_rec, scale_rec, (H, H))).astype(np.float32)
    b_h = np.zeros(H, dtype=np.float32)
    W_out = (rng.uniform(-scale_out, scale_out, (C, H))).astype(np.float32)  # (C, H) for easier matmul
    b_out = np.zeros(C, dtype=np.float32)
    params = [E, W_hh, b_h, W_out, b_out]

    # Adam state
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training data
        perm = rng.permutation(len(X_train))
        for s in range(0, len(X_train), BATCH):
            batch_idx = perm[s:s + BATCH]
            Xb, yb = X_train[batch_idx], y_train[batch_idx]

            # Forward
            logits, hiddens = rnn_forward(params, Xb)
            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # Backward
            grads = rnn_backward(params, Xb, yb, hiddens, dlogits)

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = adam_m[i], adam_v[i]
                m *= beta1; m += (1 - beta1) * g
                v *= beta2; v += (1 - beta2) * g * g
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation
        val_acc = evaluate(params, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}", file=sys.stderr)

    # Retrain on full data for best_epoch epochs
    params = [np.zeros_like(p) for p in best_params]  # re-initialize
    # Re-initialize with same seed to get same initial values
    rng = np.random.default_rng(SEED)
    E = (rng.uniform(-scale_emb, scale_emb, (256, H))).astype(np.float32)
    W_hh = (rng.uniform(-scale_rec, scale_rec, (H, H))).astype(np.float32)
    b_h = np.zeros(H, dtype=np.float32)
    W_out = (rng.uniform(-scale_out, scale_out, (C, H))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)
    params = [E, W_hh, b_h, W_out, b_out]
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(1, best_epoch + 1):
        perm = rng.permutation(N)
        for s in range(0, N, BATCH):
            batch_idx = perm[s:s + BATCH]
            Xb, yb = X[batch_idx], y[batch_idx]
            logits, hiddens = rnn_forward(params, Xb)
            dlogits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            grads = rnn_backward(params, Xb, yb, hiddens, dlogits)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = adam_m[i], adam_v[i]
                m *= beta1; m += (1 - beta1) * g
                v *= beta2; v += (1 - beta2) * g * g
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights
    E, W_hh, b_h, W_out, b_out = params
    # W_out is (C, H), need to store as (H, C) for infer.c? We'll store as (H, C) to match infer.c expectation.
    W_out_T = W_out.T.astype(np.float32)  # (H, C)
    magic = 0x524E4E31  # 'RNN1'
    header = struct.pack("<4i", magic, d, H, C)
    body = (E.astype("<f4").tobytes() +
            W_hh.astype("<f4").tobytes() +
            b_h.astype("<f4").tobytes() +
            W_out_T.astype("<f4").tobytes() +
            b_out.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-rnn inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64). */
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
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x524E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t expected = 16
        + (size_t)256 * H * 4
        + (size_t)H * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    const float *E = (const float *)(wb + 16);
    const float *W_hh = E + 256 * H;
    const float *b_h = W_hh + H * H;
    const float *W_out = b_h + H;          // (H, C)
    const float *b_out = W_out + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        memset(h, 0, H * sizeof(float));
        for (int t = 0; t < 16; t++) {
            unsigned char byte = xi[t];
            const float *e = E + byte * H;
            // h_new = tanh(e + W_hh * h + b_h)
            for (int j = 0; j < H; j++) {
                float sum = e[j] + b_h[j];
                for (int k = 0; k < H; k++) sum += W_hh[j * H + k] * h[k];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }
        // logits = W_out^T * h + b_out  (W_out is H x C stored row-major)
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int j = 0; j < H; j++) sum += W_out[j * C + c] * h[j];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```