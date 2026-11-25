```hypothesis
Family: label-embedmlp (embedding + one ReLU layer). Each of the 16 context positions gets a learned 8-dim embedding; concatenated to 128-dim vector, then a 64-unit ReLU layer, then 65-class linear. Embeddings capture character similarities, MLP models interactions. Should beat label-linear (0.3375) by learning distributed representations. If dev accuracy <= 0.3375 or CPU cost > 2x label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 8          # embedding dimension per position
H = 64         # hidden layer size
LR = 1e-3
L2 = 1e-4
BATCH = 128
MAX_EPOCHS = 100
PATIENCE = 5

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
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    n = X.shape[0]
    # validation split: last 20% in file order
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        batch = X_batch.shape[0]
        embedded = np.zeros((batch, d, E), dtype=np.float32)
        for pos in range(d):
            embedded[:, pos, :] = emb[pos, X_batch[:, pos], :]
        x = embedded.reshape(batch, d * E)
        h = x @ W1 + b1
        h = np.maximum(h, 0)
        out = h @ W2 + b2
        return out, h, x

    def softmax_cross_entropy_grad(logits, targets):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return probs / len(targets)

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    t = 0

    # Train on 80% with early stopping
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(train_size)
        for s in range(0, train_size, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_train[idx], y_train[idx]
            logits, h, x = forward(xb)
            grad_out = softmax_cross_entropy_grad(logits, yb)

            grad_W2 = h.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T
            grad_h[h <= 0] = 0
            grad_W1 = x.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_x = grad_h @ W1.T
            grad_x = grad_x.reshape(len(xb), d, E)
            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                np.add.at(grad_emb[pos], xb[:, pos], grad_x[:, pos, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
        if epoch - best_epoch >= PATIENCE:
            break

    # Retrain on all data for best_epoch+1 epochs
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for epoch in range(best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, x = forward(xb)
            grad_out = softmax_cross_entropy_grad(logits, yb)

            grad_W2 = h.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T
            grad_h[h <= 0] = 0
            grad_W1 = x.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_x = grad_h @ W1.T
            grad_x = grad_x.reshape(len(xb), d, E)
            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                np.add.at(grad_emb[pos], xb[:, pos], grad_x[:, pos, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    emb, W1, b1, W2, b2 = params

    # Write weights.bin: magic, d, E, H, C, then embeddings, W1, b1, W2, b2 (all float32 LE)
    out = struct.pack("<5i", 0x454D424C, d, E, H, C)
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * d * E);
    float *hid = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(vec + j * E, src, E * sizeof(float));
        }

        // hidden layer: hid = ReLU(vec @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += vec[v] * W1[(long)v * H + u];
            hid[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: logits = hid @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hid[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```