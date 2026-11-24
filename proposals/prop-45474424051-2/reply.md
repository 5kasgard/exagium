```hypothesis
Family: label-embedding-mlp. Learns a small embedding (size E) for each (position, byte) pair, sums them, then passes through one ReLU hidden layer (size H) to 65 classes. This captures byte similarities and position interactions that a pure linear model misses, while keeping inference fast due to small E and H. Hyperparameters (E, H) are chosen on the last 20% of train.in (validation split) then retrained on all data. If the best validation accuracy does not exceed label-linear's 0.3375, the family fails to improve.
```

```python
"""label-embedding-mlp: embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 5
VAL_FRAC = 0.2

# Hyperparameter grid to search on validation split
E_CHOICES = [16, 32, 64]
H_CHOICES = [16, 32, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X_idx, emb, W1, b1, W2, b2):
    """X_idx: (n, d) uint8. Returns logits (n, C)."""
    n, d = X_idx.shape
    E = emb.shape[2]
    # Sum embeddings
    acc = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        acc += emb[j, X_idx[:, j]]
    if W1 is not None:  # hidden layer
        h = np.maximum(acc @ W1 + b1, 0, dtype=np.float32)
        return h @ W2 + b2
    else:  # linear
        return acc @ W2 + b2


def train_epoch(X_idx, y, emb, W1, b1, W2, b2, rng, lr, l2, batch):
    n = X_idx.shape[0]
    order = rng.permutation(n)
    params = [emb, W1, b1, W2, b2] if W1 is not None else [emb, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for s in range(0, n, batch):
        idx = order[s:s + batch]
        xb, yb = X_idx[idx], y[idx]
        n_b = len(yb)
        # Forward
        acc = np.zeros((n_b, emb.shape[2]), dtype=np.float32)
        for j in range(emb.shape[0]):
            acc += emb[j, xb[:, j]]
        if W1 is not None:
            a1 = acc @ W1 + b1
            h = np.maximum(a1, 0)
            logits = h @ W2 + b2
        else:
            logits = acc @ W2 + b2
        # Softmax + cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n_b), yb] -= 1.0
        probs /= n_b
        # Backward
        if W1 is not None:
            gW2 = h.T @ probs + l2 * W2
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[a1 <= 0] = 0.0
            gW1 = acc.T @ gh + l2 * W1
            gb1 = gh.sum(axis=0)
            gacc = gh @ W1.T
        else:
            gW2 = acc.T @ probs + l2 * W2
            gb2 = probs.sum(axis=0)
            gacc = probs @ W2.T
        # Embedding gradients
        gemb = np.zeros_like(emb)
        for j in range(emb.shape[0]):
            np.add.at(gemb[j], xb[:, j], gacc)
        gemb += l2 * emb
        grads = [gemb, gW1, gb1, gW2, gb2] if W1 is not None else [gemb, gW2, gb2]
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = int(n * VAL_FRAC)
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E = best_H = None

    for E in E_CHOICES:
        for H in H_CHOICES:
            # Initialize
            emb = (rng.normal(0, 1, (d, 256, E)) * math.sqrt(2.0 / E)).astype(np.float32)
            if H > 0:
                W1 = (rng.normal(0, 1, (E, H)) * math.sqrt(2.0 / E)).astype(np.float32)
                b1 = np.zeros(H, np.float32)
                W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
                b2 = np.zeros(C, np.float32)
            else:
                W1 = b1 = None
                W2 = (rng.normal(0, 1, (E, C)) * math.sqrt(2.0 / E)).astype(np.float32)
                b2 = np.zeros(C, np.float32)

            # Train with early stopping on validation
            wait = 0
            local_best_acc = -1.0
            local_best_params = None
            for epoch in range(MAX_EPOCHS):
                params = train_epoch(X_train, y_train, emb, W1, b1, W2, b2, rng, LR, L2, BATCH)
                emb, W1, b1, W2, b2 = (params[0], params[1], params[2], params[3], params[4]) if H > 0 else (params[0], None, None, params[1], params[2])
                val_logits = forward(X_val, emb, W1, b1, W2, b2)
                val_acc = accuracy(val_logits, y_val)
                if val_acc > local_best_acc:
                    local_best_acc = val_acc
                    local_best_params = (emb.copy(), W1.copy() if W1 is not None else None,
                                         b1.copy() if b1 is not None else None,
                                         W2.copy(), b2.copy())
                    wait = 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break
            if local_best_acc > best_acc:
                best_acc = local_best_acc
                best_params = local_best_params
                best_E, best_H = E, H
                print(f"New best: E={E}, H={H}, val_acc={best_acc:.4f}", file=sys.stderr)

    print(f"Selected E={best_E}, H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset with best hyperparameters
    rng = np.random.default_rng(SEED)
    E, H = best_E, best_H
    emb = (rng.normal(0, 1, (d, 256, E)) * math.sqrt(2.0 / E)).astype(np.float32)
    if H > 0:
        W1 = (rng.normal(0, 1, (E, H)) * math.sqrt(2.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
    else:
        W1 = b1 = None
        W2 = (rng.normal(0, 1, (E, C)) * math.sqrt(2.0 / E)).astype(np.float32)
        b2 = np.zeros(C, np.float32)

    for epoch in range(MAX_EPOCHS):
        params = train_epoch(X, y, emb, W1, b1, W2, b2, rng, LR, L2, BATCH)
        emb, W1, b1, W2, b2 = (params[0], params[1], params[2], params[3], params[4]) if H > 0 else (params[0], None, None, params[1], params[2])

    # Save weights
    # Format: magic(0x454D424C), d, E, H, C (int32)
    # Then emb[d][256][E] float32 (row-major: d, then 256, then E)
    # Then if H>0: W1[E][H], b1[H], W2[H][C], b2[C] else: W2[E][C], b2[C]
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    if H > 0:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index 0..64). */
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
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20;
    expected += 4L * d * 256 * E;           // emb
    if (H > 0) {
        expected += 4L * E * H;             // W1
        expected += 4L * H;                 // b1
        expected += 4L * H * C;             // W2
        expected += 4L * C;                 // b2
    } else {
        expected += 4L * E * C;             // W2
        expected += 4L * C;                 // b2
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + (H > 0 ? E * H : 0);
    const float *W2 = b1 + (H > 0 ? H : 0);
    const float *b2 = W2 + (H > 0 ? H * C : E * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *hidden = H > 0 ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Zero accumulator
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        // Sum embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += eptr[e];
        }
        const float *logits;
        if (H > 0) {
            // Hidden layer: ReLU(acc * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += acc[e] * W1[(long)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                scores[c] = sum;
            }
            logits = scores;
        } else {
            // Linear: acc * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int e = 0; e < E; e++) sum += acc[e] * W2[(long)e * C + c];
                scores[c] = sum;
            }
            logits = scores;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```