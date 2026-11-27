```hypothesis
Family: label-embed-mlp (embedding MLP with early stopping). Character prediction benefits from distributed byte representations (e.g., vowels/consonants cluster), which one-hot linear models cannot capture. A small MLP on learned per-position embeddings adds minimal parameters (~45K) and compute, while modeling interactions via the hidden layer. Early stopping on the tuning split (last 20%) determines epochs, then retrain on all data. If dev accuracy ≤ 0.3375 (label-linear), the approach fails.
```
```python
"""label-embed-mlp: Embedding MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Architecture: per-position byte embeddings (16*256*E) -> concat -> ReLU(H) -> logits(C).
Adam lr=1e-3, batch=64, max_epochs=100, early stopping patience=5 on tuning split (last 20%).
Retrain on full train for best_epoch epochs. Export: magic, d, E, H, C, embeddings, W1, b1, W2, b2 (float32 LE)."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension per position
H = 64         # hidden units
SEED = 1
LR = 1e-3
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x44424D45  # 'EMBD' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of CE w.r.t logits, shape (batch, C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def forward(emb_tables, W1, b1, W2, b2, X):
    """X: (batch, 16) uint8. Returns logits (batch, C)."""
    batch = X.shape[0]
    # gather embeddings: (batch, 16, E)
    emb = emb_tables[np.arange(16), X]  # advanced indexing
    flat = emb.reshape(batch, -1)       # (batch, 16*E)
    hidden = flat @ W1 + b1
    np.maximum(hidden, 0, out=hidden)   # ReLU in-place
    logits = hidden @ W2 + b2
    return logits, hidden, flat


def backward(grad_logits, hidden, flat, W2, W1, emb_tables, X, batch):
    """Compute gradients for all params. Returns dict of gradients."""
    # grad_logits: (batch, C)
    # hidden: (batch, H) after ReLU
    # flat: (batch, 16*E)
    # W2: (H, C), W1: (16*E, H)
    grad_W2 = hidden.T @ grad_logits                    # (H, C)
    grad_b2 = grad_logits.sum(axis=0)                   # (C,)
    grad_hidden = grad_logits @ W2.T                    # (batch, H)
    grad_hidden[hidden <= 0] = 0.0                      # ReLU grad
    grad_W1 = flat.T @ grad_hidden                      # (16*E, H)
    grad_b1 = grad_hidden.sum(axis=0)                   # (H,)
    grad_flat = grad_hidden @ W1.T                      # (batch, 16*E)
    grad_emb = grad_flat.reshape(batch, 16, E)          # (batch, 16, E)
    # accumulate into emb_tables grad: (16, 256, E)
    grad_emb_tables = np.zeros_like(emb_tables, dtype=np.float32)
    np.add.at(grad_emb_tables, (np.arange(16), X), grad_emb)
    return {
        'emb': grad_emb_tables,
        'W1': grad_W1, 'b1': grad_b1,
        'W2': grad_W2, 'b2': grad_b2
    }


def adam_update(params, grads, m, v, t, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS):
    for key in params:
        g = grads[key]
        m[key] *= beta1; m[key] += (1 - beta1) * g
        v[key] *= beta2; v[key] += (1 - beta2) * (g * g)
        m_hat = m[key] / (1 - beta1 ** t)
        v_hat = v[key] / (1 - beta2 ** t)
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_loop(rng, emb_tables, W1, b1, W2, b2, X, y, tune_X, tune_y, max_epochs, patience):
    """Train on (X,y), validate on (tune_X, tune_y). Returns best_epoch, best_params."""
    n = X.shape[0]
    params = {'emb': emb_tables, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    best_acc = -1.0
    best_epoch = 0
    best_params = {k: v.copy() for k, v in params.items()}
    wait = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, hidden, flat = forward(params['emb'], params['W1'], params['b1'],
                                           params['W2'], params['b2'], xb)
            grad_logits = softmax_cross_entropy_grad(logits, yb)
            batch_grads = backward(grad_logits, hidden, flat, params['W2'], params['W1'],
                                   params['emb'], xb, len(idx))
            t += 1
            adam_update(params, batch_grads, m, v, t)

        # validate
        tune_logits, _, _ = forward(params['emb'], params['W1'], params['b1'],
                                    params['W2'], params['b2'], tune_X)
        acc = accuracy(tune_logits, tune_y)
        if acc > best_acc + 1e-6:
            best_acc = acc
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_epoch, best_params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n = X.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, X_tune = X[:train_size], X[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)

    # Initialize
    emb_tables = rng.normal(0, 0.1, (d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    # Phase 1: train on train split, tune on tune split
    best_epoch, best_params = train_loop(rng, emb_tables, W1, b1, W2, b2,
                                         X_train, y_train, X_tune, y_tune,
                                         MAX_EPOCHS, PATIENCE)

    # Phase 2: retrain on full data for best_epoch epochs
    rng2 = np.random.default_rng(SEED)  # same init
    emb_tables = rng2.normal(0, 0.1, (d, 256, E)).astype(np.float32)
    W1 = rng2.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng2.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = {'emb': emb_tables, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(1, best_epoch + 1):
        order = rng2.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, hidden, flat = forward(params['emb'], params['W1'], params['b1'],
                                           params['W2'], params['b2'], xb)
            grad_logits = softmax_cross_entropy_grad(logits, yb)
            batch_grads = backward(grad_logits, hidden, flat, params['W2'], params['W1'],
                                   params['emb'], xb, len(idx))
            t += 1
            adam_update(params, batch_grads, m, v, t)

    # Pack weights
    # Order: magic, d, E, H, C; then embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    out = bytearray()
    out += header
    out += params['emb'].astype("<f4").tobytes()
    out += params['W1'].astype("<f4").tobytes()
    out += params['b1'].astype("<f4").tobytes()
    out += params['W2'].astype("<f4").tobytes()
    out += params['b2'].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x44424D45), d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, src, E * sizeof(float));
        }

        // hidden = ReLU(flat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += flat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

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