```hypothesis
We propose "label-emb", a factorized linear model that learns a small embedding (size D) for each of the 16 byte positions, then applies a single linear layer to predict the next character. This captures inter-character similarities (e.g., letters vs punctuation) while drastically reducing parameters compared to one-hot linear (16*256*D + 16*D*65 vs 16*256*65). On the tiny Shakespeare dataset, such parameter sharing should improve generalization over label-linear. If dev accuracy does not exceed label-linear's 0.3375, the family does not justify its extra complexity.
```

```python
"""label-emb: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: int32 magic 0x4C424D45, d, D, C; float32 E[d][256][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (tuned on last 20% of train.in)
D_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
SEED = 1
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

def train_eval(X, y, d, C, D, lr, rng, val_idx=None):
    """Train on X,y (optionally only on train_idx) and return validation accuracy if val_idx provided."""
    n = X.shape[0]
    # Initialize embeddings and linear weights
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        emb = np.zeros((B, d * D), np.float32)
        for j in range(d):
            emb[:, j*D:(j+1)*D] = E[j, X_batch[:, j]]
        logits = emb @ W + b
        return logits, emb

    def loss_grad(logits, y_batch):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        grad_logits = probs / B
        loss = -np.log(np.maximum(probs[np.arange(B), y_batch] + 1.0, 1e-12)).mean()
        return loss, grad_logits.astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, emb = forward(Xb)
            _, grad_logits = loss_grad(logits, yb)
            # Gradients
            grad_W = emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(0)
            grad_E = np.zeros_like(E)
            for j in range(d):
                grad_emb_j = grad_logits @ W[j*D:(j+1)*D].T
                np.add.at(grad_E[j], Xb[:, j], grad_emb_j)
            grad_E += L2 * E
            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    if val_idx is not None:
        Xv, yv = X[val_idx], y[val_idx]
        logits_v, _ = forward(Xv)
        pred = logits_v.argmax(axis=1)
        return (pred == yv).mean()
    return params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_D = None
    best_lr = None

    for D in D_CHOICES:
        for lr in LR_CHOICES:
            # Use a fresh RNG stream for each config to keep deterministic
            cfg_rng = np.random.default_rng(SEED + D * 1000 + int(lr * 10000))
            # Train on train_idx only
            X_train, y_train = X[train_idx], y[train_idx]
            # We need a temporary training function that works on subset
            # We'll just call train_eval with the subset and validate on val_idx
            # But train_eval expects full X,y and val_idx. Let's adapt:
            # We'll create a temporary dataset of size len(train_idx) and train on it, then evaluate on val_idx.
            # However, we need to keep the same parameter initialization scheme. We'll just run the training loop on the subset.
            # For simplicity, we'll reuse the training code but on the subset.
            # We'll copy the training logic here to avoid confusion.
            n_tr = len(train_idx)
            E = (cfg_rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
            W = (cfg_rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
            b = np.zeros(C, np.float32)
            params = [E, W, b]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0

            def forward_batch(Xb):
                B = Xb.shape[0]
                emb = np.zeros((B, d * D), np.float32)
                for j in range(d):
                    emb[:, j*D:(j+1)*D] = E[j, Xb[:, j]]
                logits = emb @ W + b
                return logits, emb

            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
            for _ in range(epochs):
                order = cfg_rng.permutation(n_tr)
                for s in range(0, n_tr, BATCH):
                    idx = order[s:s + BATCH]
                    Xb, yb = X_train[idx], y_train[idx]
                    logits, emb = forward_batch(Xb)
                    logits = logits - logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yb)), yb] -= 1.0
                    grad_logits = (probs / len(yb)).astype(np.float32)
                    grad_W = emb.T @ grad_logits + L2 * W
                    grad_b = grad_logits.sum(0)
                    grad_E = np.zeros_like(E)
                    for j in range(d):
                        grad_emb_j = grad_logits @ W[j*D:(j+1)*D].T
                        np.add.at(grad_E[j], Xb[:, j], grad_emb_j)
                    grad_E += L2 * E
                    grads = [grad_E, grad_W, grad_b]
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            # Evaluate on validation set
            Xv, yv = X[val_idx], y[val_idx]
            logits_v, _ = forward_batch(Xv)
            pred = logits_v.argmax(axis=1)
            acc = (pred == yv).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (E.copy(), W.copy(), b.copy())
                best_D = D
                best_lr = lr

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    D = best_D
    lr = best_lr
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward_batch(Xb):
        B = Xb.shape[0]
        emb = np.zeros((B, d * D), np.float32)
        for j in range(d):
            emb[:, j*D:(j+1)*D] = E[j, Xb[:, j]]
        logits = emb @ W + b
        return logits, emb

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, emb = forward_batch(Xb)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            grad_W = emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(0)
            grad_E = np.zeros_like(E)
            for j in range(d):
                grad_emb_j = grad_logits @ W[j*D:(j+1)*D].T
                np.add.at(grad_E[j], Xb[:, j], grad_emb_j)
            grad_E += L2 * E
            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x4C424D45  # "EMBL" in little-endian
    E_flat = E.astype("<f4").tobytes()
    W_flat = W.astype("<f4").tobytes()
    b_flat = b.astype("<f4").tobytes()
    header = struct.pack("<4i", magic, d, D, C)
    out = header + E_flat + W_flat + b_flat
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weights format: int32 magic(0x4C424D45), d, D, C; float32 E[d][256][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d <= 0 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* Build concatenated embedding */
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            memcpy(emb + j * D, e, D * sizeof(float));
        }
        /* Linear layer: scores = emb * W + b */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < d * D; f++) {
            float val = emb[f];
            if (val == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += val * w[c];
        }
        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```