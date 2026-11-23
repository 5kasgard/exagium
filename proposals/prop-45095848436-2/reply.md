```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Uses shared token embeddings (65x32) for each of 16 context positions, concatenated (512-d), then ReLU hidden layer (128 units), then 65-class output. Trained with Adam (lr=1e-3, L2=1e-4) on 80% train, validated on last 20% to pick epoch, then retrained on full data. Expects >35% dev accuracy (beating label-linear's 33.75%) with similar inference cost (~10k Ir/item). Failure: accuracy <=34% or inference >2x label-linear.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
E = 32          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100   # early stopping on validation will pick best epoch

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 16 and C == 65

    # Validation split: last 20% in file order
    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)

    def init_params():
        emb = (rng.uniform(-1, 1, (C, E)) * math.sqrt(6.0 / C)).astype(np.float32)
        W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [emb, W1, b1, W2, b2]

    params = init_params()
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        batch_size = X_batch.shape[0]
        x_emb = params[0][X_batch].reshape(batch_size, d * E)   # (B, d*E)
        h = np.maximum(x_emb @ params[1] + params[2], 0, dtype=np.float32)  # ReLU
        logits = h @ params[3] + params[4]
        return x_emb, h, logits

    def loss_grad(logits, y_batch):
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_batch)), y_batch] -= 1.0
        grad_logits = (probs / len(y_batch)).astype(np.float32)
        return grad_logits

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    step = 0

    # Phase 1: train on 80%, validate on 20% to choose epoch
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(train_size)
        for s in range(0, train_size, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_train[idx], y_train[idx]
            x_emb, h, logits = forward(xb)
            grad_logits = loss_grad(logits, yb)

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * params[3]
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ params[3].T
            grad_h[h <= 0] = 0
            grad_W1 = x_emb.T @ grad_h + L2 * params[1]
            grad_b1 = grad_h.sum(axis=0)
            grad_x_emb = grad_h @ params[1].T
            grad_x_emb = grad_x_emb.reshape(len(xb), d, E)
            grad_emb = np.zeros_like(params[0])
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_x_emb[:, j, :])
            grad_emb += L2 * params[0]

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation
        val_preds = []
        for s in range(0, val_size, BATCH):
            xb = X_val[s:s + BATCH]
            _, _, logits = forward(xb)
            val_preds.append(logits.argmax(axis=1))
        val_acc = (np.concatenate(val_preds) == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}, best={best_val_acc:.4f} at epoch {best_epoch}", file=sys.stderr)

    # Phase 2: retrain on full data for best_epoch epochs
    rng = np.random.default_rng(SEED)
    params = init_params()
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    step = 0

    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            x_emb, h, logits = forward(xb)
            grad_logits = loss_grad(logits, yb)

            grad_W2 = h.T @ grad_logits + L2 * params[3]
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ params[3].T
            grad_h[h <= 0] = 0
            grad_W1 = x_emb.T @ grad_h + L2 * params[1]
            grad_b1 = grad_h.sum(axis=0)
            grad_x_emb = grad_h @ params[1].T
            grad_x_emb = grad_x_emb.reshape(len(xb), d, E)
            grad_emb = np.zeros_like(params[0])
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_x_emb[:, j, :])
            grad_emb += L2 * params[0]

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic, d, E, H, C, then emb, W1, b1, W2, b2 (all float32 LE)
    magic = 0x454D424C  # "LMBE"
    out = struct.pack("<5i", magic, d, E, H, C)
    for p in params:
        out += p.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20 + 4L * ((long)C * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)C * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
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

    free(concat); free(hidden); free(logits);
    return 0;
}
```