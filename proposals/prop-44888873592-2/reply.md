```hypothesis
We propose "label-embedding-mlp": a two-layer MLP with per-position byte embeddings (size E) followed by a ReLU hidden layer (size H) and a linear output. This captures pairwise interactions between context positions via the hidden layer, while the embedding layer reduces parameter count compared to the full one-hot linear model (4096×65). We tune (E,H) on the last 20% of train.in; a good result would be dev accuracy >0.34 with similar or lower ns/item than label-linear. If dev accuracy ≤0.33 or ns/item >2× label-linear, the approach fails.
```

```python
"""label-embedding-mlp: per-position byte embeddings -> ReLU hidden -> linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes (embed_dim, hidden_dim) on last 20% of train.in, then retrains on all data.
Exports custom binary format: magic, d, E, H, C, then embeddings[d][256][E], W1[(d*E)][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Tuning candidates: (embed_dim, hidden_dim)
CANDIDATES = [(16, 64), (32, 128), (64, 256), (32, 256)]
SEED = 1
LR, L2, BATCH, MAX_EPOCHS_TUNE, MAX_EPOCHS_FULL = 3e-3, 1e-4, 64, 15, 60
VAL_FRAC = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_model(X, y, val_X, val_y, d, C, E, H, rng, epochs, batch, lr, l2):
    n = X.shape[0]
    # Embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # Hidden: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Flatten for Adam state
    flat = np.concatenate([p.ravel() for p in params])
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    t = 0

    # Precompute validation logits function
    def forward(X_batch, emb, W1, b1, W2, b2):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup
        flat_emb = np.empty((B, d * E), dtype=np.float32)
        for j in range(d):
            flat_emb[:, j * E:(j + 1) * E] = emb[j, X_batch[:, j]]
        # Hidden
        hidden = flat_emb @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU
        # Output
        logits = hidden @ W2 + b2
        return logits

    def loss_and_grad(X_batch, y_batch):
        B = X_batch.shape[0]
        # Forward with intermediate storage for backward
        flat_emb = np.empty((B, d * E), dtype=np.float32)
        for j in range(d):
            flat_emb[:, j * E:(j + 1) * E] = emb[j, X_batch[:, j]]
        hidden_pre = flat_emb @ W1 + b1
        hidden = np.maximum(hidden_pre, 0)
        logits = hidden @ W2 + b2
        # Softmax cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B
        # Gradients
        grad_W2 = hidden.T @ probs + l2 * W2
        grad_b2 = probs.sum(axis=0)
        grad_hidden = probs @ W2.T
        grad_hidden[hidden_pre <= 0] = 0.0  # ReLU gradient
        grad_W1 = flat_emb.T @ grad_hidden + l2 * W1
        grad_b1 = grad_hidden.sum(axis=0)
        # Embedding gradients
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            idx = X_batch[:, j]
            # grad_emb[j, idx, :] += grad_hidden[:, j*E:(j+1)*E]
            # Use scatter add via loop (B small)
            for b in range(B):
                grad_emb[j, idx[b]] += grad_hidden[b, j * E:(j + 1) * E]
        grad_emb += l2 * emb
        # Flatten gradients
        grad_flat = np.concatenate([g.ravel() for g in [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]])
        loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()
        return loss, grad_flat

    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            loss, grad = loss_and_grad(Xb, yb)
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            flat -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            # Unpack flat back into params
            offset = 0
            for i, p in enumerate(params):
                sz = p.size
                p[:] = flat[offset:offset + sz].reshape(shapes[i])
                offset += sz
        # Validation
        val_logits = forward(val_X, emb, W1, b1, W2, b2)
        val_acc = accuracy(val_logits, val_y)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for validation
    val_size = int(n * VAL_FRAC)
    train_X, val_X = X[:-val_size], X[-val_size:]
    train_y, val_y = y[:-val_size], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_config = None
    best_val_acc = -1.0
    best_tuned_params = None

    for E, H in CANDIDATES:
        print(f"Tuning E={E}, H={H}", file=sys.stderr)
        params, val_acc = train_model(train_X, train_y, val_X, val_y, d, C, E, H, rng,
                                      MAX_EPOCHS_TUNE, BATCH, LR, L2)
        print(f"  val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = (E, H)
            best_tuned_params = params

    E, H = best_config
    print(f"Best config: E={E}, H={H} (val_acc={best_val_acc:.4f})", file=sys.stderr)
    # Retrain on full data
    rng_full = np.random.default_rng(SEED)  # fresh seed for final training
    final_params, _ = train_model(X, y, X, y, d, C, E, H, rng_full,
                                  MAX_EPOCHS_FULL, BATCH, LR, L2)
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x454D424C  # "LBME"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic(0x454D424C), d, E, H, C (int32 LE), then float32 LE arrays:
 *   emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = (float *)malloc(sizeof(float) * d * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!flat || !hidden || !logits) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, src, sizeof(float) * E);
        }
        // Hidden layer: hidden = ReLU(flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) {
                sum += flat[k] * W1[(long)k * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer: logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```