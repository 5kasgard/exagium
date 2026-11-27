```hypothesis
Family: label-embmlp (per-position embeddings + one ReLU layer).
Why: One-hot linear model (label-linear) uses 4096 input features. Replacing them with learned embeddings (E=4..16) reduces input dimension to 16*E, allowing a larger hidden layer (H=16..128) at lower compute. Embeddings capture character similarities; the hidden layer models non-linear n-gram interactions. Hyperparameters (E, H) are chosen on the last 20% of train.in (tuning split), then model is retrained on all train data.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item exceeds label-linear.
```

```python
"""label-embmlp: per-position embeddings + one ReLU layer for next-char prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only.
Tunes embedding dim E and hidden size H on last 20% of train.in (file order),
then retrains on full train set. Exports weights as flat float32 arrays.
Format: int32 magic=0x454D424D, d=16, E, H, C=65; then float32
emb[16][256][E], W1[16*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on tuning split
E_CHOICES = [4, 8, 16]
H_CHOICES = [16, 32, 64, 128]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_val(X, y, val_frac=0.2):
    """Split by file order: last val_frac for validation."""
    n = X.shape[0]
    val_size = int(n * val_frac)
    if val_size == 0:
        val_size = 1
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]
    return train_X, train_y, val_X, val_y


def init_params(E, H, C, rng):
    # embeddings: 16 * 256 * E
    emb = (rng.uniform(-1, 1, (16, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    # hidden layer: input size = 16*E
    in_size = 16 * E
    W1 = (rng.uniform(-1, 1, (in_size, H)) * math.sqrt(6.0 / in_size)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X_idx):
    """X_idx: (batch, 16) uint8. Returns logits (batch, C)."""
    batch = X_idx.shape[0]
    # gather embeddings: (batch, 16, E)
    embs = emb[X_idx]  # (batch, 16, E)
    flat = embs.reshape(batch, -1)  # (batch, 16*E)
    h = flat @ W1 + b1
    h = np.maximum(h, 0)  # ReLU
    logits = h @ W2 + b2
    return logits, flat, h


def softmax_grad(logits, y):
    """Return gradient of cross-entropy wrt logits: (P - one_hot) / batch."""
    batch = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch


def train_model(X, y, E, H, C, epochs, rng):
    """Train on full dataset (X,y) with given hyperparams. Returns params."""
    emb, W1, b1, W2, b2 = init_params(E, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            logits, flat, h = forward(emb, W1, b1, W2, b2, xi)
            g_logits = softmax_grad(logits, yi)  # (batch, C)
            # grad W2, b2
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            # grad h
            g_h = g_logits @ W2.T
            g_h[h <= 0] = 0  # ReLU grad
            # grad W1, b1
            g_W1 = flat.T @ g_h + L2 * W1
            g_b1 = g_h.sum(axis=0)
            # grad flat -> grad embeddings
            g_flat = g_h @ W1.T  # (batch, 16*E)
            g_embs = g_flat.reshape(-1, 16, E)  # (batch, 16, E)
            # accumulate per position
            g_emb = np.zeros_like(emb)
            for pos in range(16):
                # xi[:, pos] are indices 0..255
                np.add.at(g_emb[pos], xi[:, pos], g_embs[:, pos])
            g_emb += L2 * emb
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def eval_accuracy(X, y, params):
    emb, W1, b1, W2, b2 = params
    logits, _, _ = forward(emb, W1, b1, W2, b2, X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def pack_weights(params, d, E, H, C):
    emb, W1, b1, W2, b2 = params
    # flatten in order
    out = bytearray()
    out += struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    return bytes(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    # Split for tuning (last 20% by file order)
    train_X, train_y, val_X, val_y = split_train_val(X, y, 0.2)

    # Grid search
    best_acc = -1.0
    best_E = best_H = None
    best_params = None
    rng_base = np.random.default_rng(SEED)
    for E in E_CHOICES:
        for H in H_CHOICES:
            # Use same seed for each config to compare fairly
            rng = np.random.default_rng(SEED)
            n_train = train_X.shape[0]
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
            params = train_model(train_X, train_y, E, H, C, epochs, rng)
            acc = eval_accuracy(val_X, val_y, params)
            print(f"E={E}, H={H}, val_acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc or (acc == best_acc and E * H < best_E * best_H):
                best_acc = acc
                best_E, best_H = E, H
                best_params = params

    print(f"Best: E={best_E}, H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparams
    rng = np.random.default_rng(SEED)
    n_full = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    final_params = train_model(X, y, best_E, best_H, C, epochs, rng)

    # Write weights
    weights = pack_weights(final_params, d, best_E, best_H, C)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output: one byte class index (0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D424D, d, E, H, C
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H], b1[H]
 *   float32 W2[H][C], b2[C]
 * Computes: emb lookup -> concat -> ReLU(W1*x+b1) -> W2*h+b2 -> argmax. */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E < 1 || E > 256 || H < 1 || H > 1024) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
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

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // hidden layer: ReLU(W1 * concat + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            const float *w = W1 + (long)u * d * E;
            for (int v = 0; v < d * E; v++) sum += concat[v] * w[v];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: W2 * hidden + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
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