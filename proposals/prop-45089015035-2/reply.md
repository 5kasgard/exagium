```hypothesis
Family: label-embmlp. A CBOW-style model with per-position byte embeddings (D=16) summed and passed through a ReLU hidden layer (H=64) to predict next character. Captures pairwise interactions via the hidden layer while keeping inference at ~5k FLOPs/item. Should beat label-linear (0.3375) on dev; if it doesn't, the hidden layer adds no useful nonlinearity for this task.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split in a prior run)
D = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, Z, y):
    emb, W1, b1, W2, b2 = params
    n = Z.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        z = Z[s:s+1024]
        yy = y[s:s+1024]
        acc = z @ emb.reshape(-1, D)  # (batch, D) -- but z is one-hot? We'll compute differently.
    # We'll implement a dedicated forward pass for accuracy.
    return 0.0  # placeholder


def forward_batch(emb, W1, b1, W2, b2, X_batch):
    """X_batch: (batch, 16) uint8 indices"""
    batch = X_batch.shape[0]
    # Sum embeddings: for each position j, add emb[j, X_batch[:, j]]
    acc = np.zeros((batch, D), dtype=np.float32)
    for j in range(16):
        acc += emb[j, X_batch[:, j]]
    # Hidden layer with ReLU
    hidden = acc @ W1 + b1
    hidden = np.maximum(hidden, 0)
    # Output layer
    scores = hidden @ W2 + b2
    return scores


def train_on_split(X, y, emb, W1, b1, W2, b2, rng, n_epochs):
    n = X.shape[0]
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            batch = Xb.shape[0]

            # Forward
            acc = np.zeros((batch, D), dtype=np.float32)
            for j in range(16):
                acc += emb[j, Xb[:, j]]
            hidden_pre = acc @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            scores = hidden @ W2 + b2

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(batch), yb] -= 1.0
            probs /= batch  # gradient of loss w.r.t scores

            # Backprop
            grad_W2 = hidden.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_hidden = probs @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = acc.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_acc = grad_hidden @ W1.T

            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(16):
                # For each example, add grad_acc to emb[j, Xb[:, j]]
                np.add.at(grad_emb[j], Xb[:, j], grad_acc)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)


def evaluate(emb, W1, b1, W2, b2, X, y):
    scores = forward_batch(emb, W1, b1, W2, b2, X)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Grid search over D, H (small grid)
    best_acc = -1.0
    best_params = None
    best_D, best_H = D, H

    for D_try in [8, 16]:
        for H_try in [32, 64]:
            rng = np.random.default_rng(SEED)
            # Initialize
            emb = (rng.normal(0, 1, (16, 256, D_try)) * math.sqrt(1.0 / D_try)).astype(np.float32)
            W1 = (rng.normal(0, 1, (D_try, H_try)) * math.sqrt(2.0 / D_try)).astype(np.float32)
            b1 = np.zeros(H_try, np.float32)
            W2 = (rng.normal(0, 1, (H_try, C)) * math.sqrt(2.0 / H_try)).astype(np.float32)
            b2 = np.zeros(C, np.float32)

            # Train on train split for 50 epochs
            train_on_split(X_train, y_train, emb, W1, b1, W2, b2, rng, 50)
            acc = evaluate(emb, W1, b1, W2, b2, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_D, best_H = D_try, H_try
                best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 1, (16, 256, best_D)) * math.sqrt(1.0 / best_D)).astype(np.float32)
    W1 = (rng.normal(0, 1, (best_D, best_H)) * math.sqrt(2.0 / best_D)).astype(np.float32)
    b1 = np.zeros(best_H, np.float32)
    W2 = (rng.normal(0, 1, (best_H, C)) * math.sqrt(2.0 / best_H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_on_split(X, y, emb, W1, b1, W2, b2, rng, n_epochs_full)

    # Write weights
    # Magic: 0x454D424C ('LMBE' little-endian)
    magic = 0x454D424C
    header = struct.pack("<5i", magic, 16, best_D, best_H, C)
    # Embeddings: 16 * 256 * D floats, row-major (pos, byte, dim)
    emb_flat = emb.transpose(0, 1, 2).reshape(-1).astype("<f4")
    W1_flat = W1.astype("<f4").ravel()
    b1_flat = b1.astype("<f4").ravel()
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4").ravel()
    out = header + emb_flat.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding MLP (CBOW + one ReLU layer) for next-char prediction.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Expected weight size: header(20) + emb(d*256*D) + W1(D*H) + b1(H) + W2(H*C) + b2(C) floats
    long expected = 20 + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 5);
    const float *emb = ptr; ptr += (long)d * 256 * D;
    const float *W1 = ptr; ptr += (long)D * H;
    const float *b1 = ptr; ptr += H;
    const float *W2 = ptr; ptr += (long)H * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int j = 0; j < D; j++) acc[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) acc[j] += e[j];
        }

        // Hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < D; j++) sum += acc[j] * W1[(long)j * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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