```hypothesis
Family: label-emb-mlp (embedding MLP with one hidden layer of 128 units).  
Why: The linear model treats each position independently; a hidden layer can capture interactions between the 16 context bytes, which is crucial for character-level language modeling. Embedding lookup is cheap, and a 128-unit ReLU layer adds modest compute (≈5k multiply-adds/item).  
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds 2× label-linear, the extra capacity does not help or overfits.
```

```python
"""label-emb-mlp: Embedding MLP for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Architecture: 16 positions × 256 vocab → 128-dim embeddings (summed) → ReLU → 65 logits.
Training: Adam lr=1e-3, L2=1e-5, batch=64, epochs chosen on last 20% validation split.
Weights format: magic 0x4E455431, d, H, C; float32 embed[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 50
VAL_FRAC = 0.2  # last 20% for validation
H = 128  # hidden size


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    n = len(X_bytes)
    assert n == len(y_bytes)

    # Convert to numpy arrays
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    # Split: last 20% for validation (tuning split)
    val_start = int(n * (1 - VAL_FRAC))
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: d x 256 x H
    scale = math.sqrt(6.0 / (256 + H))
    embed = (rng.uniform(-scale, scale, (d, 256, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    scale2 = math.sqrt(6.0 / (H + C))
    W2 = (rng.uniform(-scale2, scale2, (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embed, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(X_batch, embed, b1, W2, b2):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Sum embeddings for each position
        hidden = np.zeros((batch, H), dtype=np.float32)
        for j in range(d):
            hidden += embed[j, X_batch[:, j]]
        hidden += b1
        np.maximum(hidden, 0, out=hidden)  # ReLU
        logits = hidden @ W2 + b2
        return logits, hidden

    def loss_grad(logits, y_batch, hidden):
        batch = logits.shape[0]
        # Softmax + cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        probs /= batch
        # Gradients
        grad_W2 = hidden.T @ probs
        grad_b2 = probs.sum(axis=0)
        grad_hidden = probs @ W2.T
        grad_hidden[hidden == 0] = 0.0  # ReLU derivative
        # Embedding gradients: accumulate per position
        grad_embed = np.zeros_like(embed)
        for j in range(d):
            idx = X_batch[:, j]
            np.add.at(grad_embed[j], idx, grad_hidden)
        grad_b1 = grad_hidden.sum(axis=0)
        return [grad_embed, grad_b1, grad_W2, grad_b2]

    best_val_acc = -1.0
    best_params = None
    best_epoch = 0

    # Training loop with validation
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            logits, hidden = forward(X_batch, embed, b1, W2, b2)
            grads = loss_grad(logits, y_batch, hidden)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)
                # L2 regularization (weight decay)
                if i != 1 and i != 3:  # not biases
                    p -= LR * L2 * p

        # Validation
        val_logits, _ = forward(X_val, embed, b1, W2, b2)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}", file=sys.stderr)

    # Retrain on full dataset for best_epoch epochs
    embed, b1, W2, b2 = best_params
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            logits, hidden = forward(X_batch, embed, b1, W2, b2)
            grads = loss_grad(logits, y_batch, hidden)
            t += 1
            for i, (p, gr) in enumerate(zip([embed, b1, W2, b2], grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)
                if i != 1 and i != 3:
                    p -= LR * L2 * p

    # Write weights
    magic = 0x4E455431  # "NET1"
    out = struct.pack("<4i", magic, d, H, C)
    out += embed.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4E455431, d, H, C; float32 embed[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4E455431) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0 || C <= 0 || C > 256) {
        fprintf(stderr, "bad weights dimensions\n"); return 3;
    }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 4);
    const float *b1 = embed + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!hidden || !logits) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        memcpy(hidden, b1, sizeof(float) * H);
        // add embeddings
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += e[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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