```hypothesis
Family: label-emb (per-position embeddings + 1-hidden-layer MLP).
Why: The task predicts the next character from 16 raw bytes. One-hot linear models (label-linear) ignore similarity between characters and use 4096 sparse features. Embeddings (8-dim per position) reduce parameters to ~45k, capture character relationships, and preserve positional order via concatenation. A small ReLU MLP (64 hidden) adds non-linearity. This should exceed label-linear's 33.75% accuracy while staying far faster than label-gbdt (131k Ir/item). Failure: dev accuracy ≤ 0.3375 or Ir/item ≥ 10k.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
D = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X_bytes.shape
    assert d == 16

    # Split: last 20% for tuning
    tune_size = n // 5
    train_size = n - tune_size
    X_train, X_tune = X_bytes[:train_size], X_bytes[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)

    # Embeddings: [16, 256, D]
    emb = rng.normal(0, 0.01, size=(16, 256, D)).astype(np.float32)

    # MLP weights
    W1 = rng.normal(0, math.sqrt(2.0 / (16 * D)), size=(16 * D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X, emb, W1, b1, W2, b2):
        batch = X.shape[0]
        embedded = np.zeros((batch, 16 * D), dtype=np.float32)
        for p in range(16):
            embedded[:, p * D:(p + 1) * D] = emb[p, X[:, p]]
        h = embedded @ W1 + b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
        return logits, h, embedded

    def loss_and_grad(X, y):
        batch = X.shape[0]
        logits, h, embedded = forward(X, emb, W1, b1, W2, b2)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(batch), y] + 1e-12).mean()
        dlogits = probs
        dlogits[np.arange(batch), y] -= 1.0
        dlogits /= batch
        dW2 = h.T @ dlogits + L2 * W2
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ W2.T
        dh[h <= 0] = 0
        dW1 = embedded.T @ dh + L2 * W1
        db1 = dh.sum(axis=0)
        demb = np.zeros_like(emb)
        for p in range(16):
            np.add.at(demb[p], X[:, p], dh[:, p * D:(p + 1) * D])
        demb += L2 * emb
        return loss, [demb, dW1, db1, dW2, db2]

    def evaluate(X, y):
        logits, _, _ = forward(X, emb, W1, b1, W2, b2)
        return (logits.argmax(axis=1) == y).mean()

    best_tune_acc = 0.0
    best_params = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(train_size)
        for s in range(0, train_size, BATCH):
            idx = order[s:s + BATCH]
            loss, grads = loss_and_grad(X_train[idx], y_train[idx])
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        tune_acc = evaluate(X_tune, y_tune)
        if tune_acc > best_tune_acc:
            best_tune_acc = tune_acc
            best_params = [p.copy() for p in params]
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    emb, W1, b1, W2, b2 = best_params

    # Export: magic, d, D, H, C; then emb, W1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D424C  # "LBME"
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20L + 16L * 256 * D * 4 + 16L * D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + 16 * 256 * D;
    const float *b1 = W1 + 16 * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *embedded = malloc(sizeof(float) * 16 * D);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int p = 0; p < 16; p++) {
            const float *e = emb + (p * 256 + xi[p]) * D;
            memcpy(embedded + p * D, e, D * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < 16 * D; j++) sum += embedded[j] * W1[j * H + h];
            hidden[h] = sum > 0 ? sum : 0;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```