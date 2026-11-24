```hypothesis
We propose a "shared-embedding MLP" family (label-shared-emb) that uses a single token embedding matrix (256 × E) shared across all 16 context positions, plus learned positional embeddings (16 × E). The summed embeddings are fed into one ReLU hidden layer of size H, then a linear output layer. This factorization drastically reduces parameters vs. per-position one-hot (label-linear) while capturing positional invariances, which should improve generalization on character-level language modeling. Hyperparameters E and H are chosen on the tuning split (last 20% of train). If dev accuracy does not exceed label-linear's 0.3375, the family offers no benefit.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
E_CHOICES = [8, 16, 32]
H_CHOICES = [32, 64, 128]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 30


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t. logits."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_model(X, y, X_val, y_val, E, H, C, rng, epochs):
    """Train model, return (params, val_acc)."""
    n, d = X.shape
    # Initialize parameters
    token_emb = rng.normal(0, 0.01, (256, E)).astype(np.float32)
    pos_emb = rng.normal(0, 0.01, (d, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [token_emb, pos_emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            # Embeddings: for each position, token_emb[xb[:,j]] + pos_emb[j]
            emb = np.empty((bsz, d * E), np.float32)
            for j in range(d):
                emb[:, j*E:(j+1)*E] = token_emb[xb[:, j]] + pos_emb[j]
            h_pre = emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Loss gradient
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            # Backward
            # Output layer
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)
            # Hidden layer
            dh = g @ W2.T
            dh *= (h_pre > 0).astype(np.float32)
            grad_W1 = emb.T @ dh + L2 * W1
            grad_b1 = dh.sum(axis=0)
            # Embeddings
            demb = dh @ W1.T  # (bsz, d*E)
            grad_token_emb = np.zeros_like(token_emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                dj = demb[:, j*E:(j+1)*E]
                # token_emb gradient
                np.add.at(grad_token_emb, xb[:, j], dj)
                # pos_emb gradient
                grad_pos_emb[j] = dj.sum(axis=0)

            grads = [grad_token_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        val_acc = evaluate(X_val, y_val, params, d, E, H, C)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    return best_params, best_val_acc


def evaluate(X, y, params, d, E, H, C):
    token_emb, pos_emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb, yb = X[s:s+1024], y[s:s+1024]
        bsz = len(yb)
        emb = np.empty((bsz, d * E), np.float32)
        for j in range(d):
            emb[:, j*E:(j+1)*E] = token_emb[xb[:, j]] + pos_emb[j]
        h = np.maximum(emb @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Hyperparameter search
    best_E, best_H = None, None
    best_val_acc = -1.0
    best_params = None

    for E in E_CHOICES:
        for H in H_CHOICES:
            print(f"Trying E={E}, H={H}", file=sys.stderr)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
            params, val_acc = train_model(X_train, y_train, X_val, y_val, E, H, C, rng, epochs)
            print(f"  val_acc={val_acc:.4f}", file=sys.stderr)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_E, best_H = E, H
                best_params = params

    print(f"Best: E={best_E}, H={best_H}, val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)  # reset seed for deterministic final training
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params, _ = train_model(X, y, X_val, y_val, best_E, best_H, C, rng, epochs)
    token_emb, pos_emb, W1, b1, W2, b2 = final_params

    # Build T table: for each position j (0..d-1) and byte v (0..255), compute contribution to hidden layer
    # T[j][v] = (token_emb[v] + pos_emb[j]) @ W1[j*E:(j+1)*E]   (shape H)
    T = np.empty((d, 256, best_H), dtype=np.float32)
    for j in range(d):
        block = W1[j*best_E:(j+1)*best_E]  # (E, H)
        for v in range(256):
            T[j, v] = (token_emb[v] + pos_emb[j]) @ block

    # Write weights.bin
    # Magic: 0x53454D4C ("SEML" little-endian)
    magic = 0x53454D4C
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Shared-embedding MLP inference: ./infer weights.bin in.rec N
 * Weights format: magic 0x53454D4C, d, H, C; then T[d][256][H] (float32), b1[H], W2[H][C], b2[C].
 * Input record: d bytes. Output: one byte class index. */
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
    if (nw < 16 || hd[0] != 0x53454D4C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```