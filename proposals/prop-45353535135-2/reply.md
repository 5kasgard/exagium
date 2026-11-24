```hypothesis
Family: embedding-mlp (position-wise byte embeddings + one ReLU hidden layer).
Why: label-linear uses 16 independent one-hot tables (16*256*65 ≈ 266k params) but no cross-position interaction.
An embedding layer (16*256*8 = 32k) followed by a 64-unit hidden layer (16*8*64 + 64*65 ≈ 12k) totals ~45k params,
allowing the model to learn positional interactions while staying smaller and faster than label-linear.
Failure: if dev accuracy ≤ 0.34 (not beating label-linear's 0.3375) the extra complexity is not justified.
```

```python
"""Embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Architecture: 16 positions, each with 256 -> E=8 embedding; concat -> 128-d -> ReLU(64) -> 65 logits.
Training: Adam lr=3e-3, L2=1e-4, batch=64, early stopping on last 20% tuning split, then retrain on all data.
Weights format (little-endian):
  uint32 magic=0x44424D45 ('EMBD'), d=16, E=8, H=64, C=65
  float32 emb[d][256][E]       (16*256*8)
  float32 W1[d*E][H]           (128*64)
  float32 b1[H]                (64)
  float32 W2[H][C]             (64*65)
  float32 b2[C]                (65)
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5   # early stopping patience
SAMPLES_PER_EPOCH = 600000  # not used directly; we run full epochs


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(rng, d, E, H, C):
    # embeddings: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # hidden layer: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(X, emb, W1, b1, W2, b2):
    """X: (N, d) uint8. Returns logits (N, C) and hidden pre-activation for backprop."""
    N, d = X.shape
    # embedding lookup
    cat = np.empty((N, d * E), dtype=np.float32)
    for j in range(d):
        cat[:, j * E:(j + 1) * E] = emb[j][X[:, j]]
    # hidden
    h_pre = cat @ W1 + b1
    h = np.maximum(h_pre, 0, dtype=np.float32)
    # output
    logits = h @ W2 + b2
    return logits, h_pre, h, cat


def loss_and_grad(logits, y, h_pre, h, cat, X, emb, W1, b1, W2, b2, L2):
    N = logits.shape[0]
    # softmax + cross-entropy gradient
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    dlogits = probs / N

    # output layer grads
    dW2 = h.T @ dlogits + L2 * W2
    db2 = dlogits.sum(axis=0)

    # hidden layer grads
    dh = dlogits @ W2.T
    dh[h_pre <= 0] = 0.0

    dW1 = cat.T @ dh + L2 * W1
    db1 = dh.sum(axis=0)

    # embedding grads
    dcat = dh @ W1.T
    demb = [np.zeros_like(e) for e in emb]
    for j in range(len(emb)):
        np.add.at(demb[j], X[:, j], dcat[:, j * E:(j + 1) * E])

    return dW1, db1, dW2, db2, demb


def adam_step(params, grads, states, t, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g, (m, v) in zip(params, grads, states):
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_one_pass(X, y, emb, W1, b1, W2, b2, rng, max_epochs, batch_size, tune_idx=None):
    """Returns (best_epoch, best_emb, best_W1, best_b1, best_W2, best_b2)"""
    n = X.shape[0]
    if tune_idx is not None:
        train_idx = np.arange(tune_idx)
        val_idx = np.arange(tune_idx, n)
    else:
        train_idx = np.arange(n)
        val_idx = None

    # Adam states
    params = []
    for e in emb: params.append(e)
    params.extend([W1, b1, W2, b2])
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None
    no_improve = 0
    t = 0

    for epoch in range(1, max_epochs + 1):
        # shuffle training indices
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = X[idx], y[idx]

            logits, h_pre, h, cat = forward(Xb, emb, W1, b1, W2, b2)
            dW1, db1, dW2, db2, demb = loss_and_grad(logits, yb, h_pre, h, cat, Xb, emb, W1, b1, W2, b2, L2)

            t += 1
            grads = demb + [dW1, db1, dW2, db2]
            adam_step(params, grads, states, t)

        # validation
        if val_idx is not None:
            val_logits, _, _, _ = forward(X[val_idx], emb, W1, b1, W2, b2)
            val_acc = accuracy(val_logits, y[val_idx])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_params = [p.copy() for p in params]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

    if val_idx is not None:
        # restore best
        for p, bp in zip(params, best_params):
            p[:] = bp
        return best_epoch
    else:
        return max_epochs


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # tuning split: last 20% in file order
    tune_start = int(0.8 * n)

    rng = np.random.default_rng(SEED)

    # First pass: train on first 80%, validate on last 20% to pick best epoch
    emb, W1, b1, W2, b2 = init_params(rng, d, E, H, C)
    best_epoch = train_one_pass(X, y, emb, W1, b1, W2, b2, rng, MAX_EPOCHS, BATCH, tune_start)

    # Second pass: retrain on all data for best_epoch epochs
    rng = np.random.default_rng(SEED)  # reset seed for identical initialization
    emb, W1, b1, W2, b2 = init_params(rng, d, E, H, C)
    train_one_pass(X, y, emb, W1, b1, W2, b2, rng, best_epoch, BATCH, tune_idx=None)

    # Write weights
    magic = 0x44424D45  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    # embeddings: d * 256 * E
    emb_flat = np.concatenate([e.ravel() for e in emb]).astype("<f4")
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
/* Embedding + one ReLU hidden layer inference.
 * ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index (argmax).
 * Weights format (little-endian):
 *   uint32 magic=0x44424D45, d=16, E=8, H=64, C=65
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H]
 *   float32 b1[H]
 *   float32 W2[H][C]
 *   float32 b2[C]
 */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    // expected weight size
    long exp = 20
        + (long)d * 256 * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != exp) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // buffers
    float *cat = malloc(sizeof(float) * d * E);
    float *h_pre = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(cat + j * E, e, E * sizeof(float));
        }

        // hidden layer: h_pre = cat * W1 + b1
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += cat[v] * W1[(long)v * H + u];
            h_pre[u] = sum;
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: logits = h * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(cat); free(h_pre); free(h); free(logits);
    return 0;
}
```