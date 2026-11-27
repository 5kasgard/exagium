```hypothesis
Family: label-embed (shared token embeddings + positional embeddings, one hidden ReLU layer).
Why: The 16-byte context benefits from shared character embeddings across positions (reducing parameters vs. one-hot) while positional embeddings preserve order. A small hidden layer adds non-linearity. This should beat label-linear (0.3375) by learning distributed representations, and be faster than label-gbdt.
Failure: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear significantly, the hypothesis is wrong.
```

```python
"""label-embed: shared token embeddings + positional embeddings, one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split
GRID = {
    'D': [16, 32],
    'H': [32, 64],
    'LR': [1e-3, 3e-3],
    'L2': [1e-4],
    'BATCH': [64],
    'EPOCHS_TUNE': 5,
    'EPOCHS_FULL': 20,
    'SEED': 1,
}
SAMPLES_PER_EPOCH = 600000  # max samples per epoch for full training


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def softmax_cross_entropy_grad(logits, y):
    # logits: (batch, C), y: (batch,)
    batch = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch  # gradient w.r.t logits


def forward(X, E, P, W1, b1, W2, b2):
    # X: (batch, 16) uint8
    # E: (256, D), P: (16, D)
    # returns logits (batch, C), hidden (batch, H) for backward
    batch = X.shape[0]
    D = E.shape[1]
    # concatenated embeddings: (batch, 16*D)
    concat = np.empty((batch, 16 * D), dtype=np.float32)
    for p in range(16):
        concat[:, p*D:(p+1)*D] = E[X[:, p]] + P[p]
    hidden = concat @ W1 + b1
    hidden_relu = np.maximum(hidden, 0)
    logits = hidden_relu @ W2 + b2
    return logits, concat, hidden, hidden_relu


def backward(concat, hidden, hidden_relu, grad_logits, W1, W2):
    # grad_logits: (batch, C)
    # returns grads for E, P, W1, b1, W2, b2
    batch = concat.shape[0]
    D = W1.shape[1] // 16
    H = W2.shape[0]
    C = W2.shape[1]

    grad_hidden_relu = grad_logits @ W2.T  # (batch, H)
    grad_hidden = grad_hidden_relu * (hidden > 0)  # (batch, H)

    grad_W2 = hidden_relu.T @ grad_logits  # (H, C)
    grad_b2 = grad_logits.sum(axis=0)  # (C,)

    grad_W1 = concat.T @ grad_hidden  # (16*D, H)
    grad_b1 = grad_hidden.sum(axis=0)  # (H,)

    grad_concat = grad_hidden @ W1.T  # (batch, 16*D)

    grad_E = np.zeros((256, D), dtype=np.float32)
    grad_P = np.zeros((16, D), dtype=np.float32)
    for p in range(16):
        sl = slice(p*D, (p+1)*D)
        # accumulate gradients for each token value
        # we need to scatter-add grad_concat[:, sl] to grad_E[X[:, p]]
        # simple loop over batch (batch <= 64, D <= 32, 16 positions -> fine)
        for i in range(batch):
            token = concat_i = grad_concat[i, sl]
            t = X_batch[i, p]  # need X_batch; we'll pass it separately
    # We'll restructure: pass X_batch to backward
    return grad_E, grad_P, grad_W1, grad_b1, grad_W2, grad_b2


def train_epoch(X, y, E, P, W1, b1, W2, b2, lr, l2, batch_size, rng, state):
    n = X.shape[0]
    order = rng.permutation(n)
    t = state['t']
    for s in range(0, n, batch_size):
        idx = order[s:s+batch_size]
        X_batch = X[idx]
        y_batch = y[idx]

        logits, concat, hidden, hidden_relu = forward(X_batch, E, P, W1, b1, W2, b2)
        grad_logits = softmax_cross_entropy_grad(logits, y_batch)

        # Backward
        batch = X_batch.shape[0]
        D = E.shape[1]
        H = W2.shape[0]
        C = W2.shape[1]

        grad_hidden_relu = grad_logits @ W2.T
        grad_hidden = grad_hidden_relu * (hidden > 0)

        grad_W2 = hidden_relu.T @ grad_logits + l2 * W2
        grad_b2 = grad_logits.sum(axis=0)

        grad_W1 = concat.T @ grad_hidden + l2 * W1
        grad_b1 = grad_hidden.sum(axis=0)

        grad_concat = grad_hidden @ W1.T  # (batch, 16*D)

        grad_E = np.zeros((256, D), dtype=np.float32)
        grad_P = np.zeros((16, D), dtype=np.float32)
        for p in range(16):
            sl = slice(p*D, (p+1)*D)
            for i in range(batch):
                token = X_batch[i, p]
                grad_E[token] += grad_concat[i, sl]
                grad_P[p] += grad_concat[i, sl]

        # Adam updates
        t += 1
        params = [E, P, W1, b1, W2, b2]
        grads = [grad_E, grad_P, grad_W1, grad_b1, grad_W2, grad_b2]
        for i, (p, g) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        state['t'] = t
    return state


def evaluate(X, y, E, P, W1, b1, W2, b2, batch_size=1024):
    n = X.shape[0]
    preds = np.empty(n, dtype=np.int64)
    for s in range(0, n, batch_size):
        X_batch = X[s:s+batch_size]
        logits, _, _, _ = forward(X_batch, E, P, W1, b1, W2, b2)
        preds[s:s+batch_size] = logits.argmax(axis=1)
    return accuracy(y, preds)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    d = card["input"]["bytes"]  # 16
    C = card["classes"]  # 65

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    best_acc = -1
    best_params = None
    best_state = None

    rng = np.random.default_rng(GRID['SEED'])

    for D in GRID['D']:
        for H in GRID['H']:
            for LR in GRID['LR']:
                for L2 in GRID['L2']:
                    # Initialize
                    E = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
                    P = (rng.uniform(-1, 1, (16, D)) * math.sqrt(6.0 / 16)).astype(np.float32)
                    W1 = (rng.uniform(-1, 1, (16*D, H)) * math.sqrt(6.0 / (16*D))).astype(np.float32)
                    b1 = np.zeros(H, np.float32)
                    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
                    b2 = np.zeros(C, np.float32)

                    state = {
                        't': 0,
                        0: (np.zeros_like(E), np.zeros_like(E)),
                        1: (np.zeros_like(P), np.zeros_like(P)),
                        2: (np.zeros_like(W1), np.zeros_like(W1)),
                        3: (np.zeros_like(b1), np.zeros_like(b1)),
                        4: (np.zeros_like(W2), np.zeros_like(W2)),
                        5: (np.zeros_like(b2), np.zeros_like(b2)),
                    }

                    for _ in range(GRID['EPOCHS_TUNE']):
                        state = train_epoch(X_train, y_train, E, P, W1, b1, W2, b2,
                                            LR, L2, GRID['BATCH'], rng, state)

                    acc = evaluate(X_tune, y_tune, E, P, W1, b1, W2, b2)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (E.copy(), P.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())
                        best_state = state

    print(f"Tuned: D={best_params[0].shape[1]}, H={best_params[2].shape[1]}, LR={LR}, L2={L2}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters (using same LR, L2, etc.)
    E, P, W1, b1, W2, b2 = best_params
    state = best_state
    # Reset Adam state? Continue from tune state? We'll continue for more epochs.
    full_epochs = GRID['EPOCHS_FULL']
    for _ in range(full_epochs):
        state = train_epoch(X_all, y_all, E, P, W1, b1, W2, b2,
                            LR, L2, GRID['BATCH'], rng, state)

    # Export weights
    # Format: magic, d, D, H, C, then float32 arrays: E, P, W1, b1, W2, b2
    D = E.shape[1]
    H = W1.shape[1]
    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
    out += E.astype("<f4").tobytes()
    out += P.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: one byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected = 20L + 4L * (256L * D + 16L * D + 16L * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *P = E + 256 * D;
    const float *W1 = P + 16 * D;
    const float *b1 = W1 + 16 * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * 16 * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embeddings
        for (int p = 0; p < 16; p++) {
            const float *e = E + xi[p] * D;
            const float *pp = P + p * D;
            float *out = concat + p * D;
            for (int j = 0; j < D; j++) out[j] = e[j] + pp[j];
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h; // column-major? W1 is (16*D, H) row-major: W1[row*H + col]
            for (int j = 0; j < 16 * D; j++) sum += concat[j] * W1[j * H + h];
            hidden[h] = sum > 0 ? sum : 0;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
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