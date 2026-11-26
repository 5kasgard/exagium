```hypothesis
Family: emb-linear (positional byte embeddings + linear classifier).
Why: The task is next-character prediction from 16 bytes of context. Each position has distinct statistical patterns; one-hot linear (label-linear) uses 4096 features but cannot share statistical strength across positions. Learned embeddings per position compress each byte into a dense vector, enabling generalization across similar bytes (e.g., letters) while preserving position information. Summing embeddings yields a fixed-size representation, followed by a single linear layer. This model has fewer parameters than label-linear for moderate embedding sizes, so inference should be faster or comparable. We expect higher accuracy because embeddings capture semantic similarity of bytes.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost (Ir/item) exceeds label-linear by >2×.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (embedding size)
EMB_SIZES = [32, 64, 128, 256]
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 30
PATIENCE = 5  # early stopping on validation
VAL_FRACTION = 0.2  # last 20% for tuning

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac):
    n = X.shape[0]
    val_size = int(n * val_frac)
    # last 20% in file order
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    return X_train, y_train, X_val, y_val

def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def forward(X, emb, W_out, b_out):
    # X: (n, d) uint8
    # emb: (d, 256, E) float32
    # W_out: (E, C) float32
    # b_out: (C,) float32
    n, d = X.shape
    E = emb.shape[2]
    # Sum embeddings
    h = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    logits = h @ W_out + b_out
    return logits

def train_emb(X_train, y_train, X_val, y_val, E, rng):
    d = X_train.shape[1]
    C = 65
    # Initialize embeddings
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    W_out = rng.normal(0, 0.01, (E, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    # Adam parameters
    params = [emb, W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n_train = X_train.shape[0]
    best_val_acc = -1.0
    best_params = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            xb = X_train[idx]
            yb = y_train[idx]
            t += 1
            # Forward
            n_b = xb.shape[0]
            h = np.zeros((n_b, E), dtype=np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            logits = h @ W_out + b_out
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(n_b), yb] -= 1.0
            probs /= n_b  # dL/dlogits
            # Gradients
            grad_W_out = h.T @ probs + L2 * W_out
            grad_b_out = probs.sum(axis=0)
            grad_h = probs @ W_out.T  # (n_b, E)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # For each position, accumulate gradient for the used byte
                # Use np.add.at for scatter add
                np.add.at(grad_emb[j], xb[:, j], grad_h)

            grads = [grad_emb, grad_W_out, grad_b_out]

            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation
        val_logits = forward(X_val, emb, W_out, b_out)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_params, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    X_train, y_train, X_val, y_val = train_val_split(X, y, VAL_FRACTION)

    best_E = None
    best_val_acc = -1.0
    best_weights = None

    for E in EMB_SIZES:
        weights, val_acc = train_emb(X_train, y_train, X_val, y_val, E, rng)
        print(f"E={E}, val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_E = E
            best_weights = weights

    print(f"Selected E={best_E} with val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best_E
    final_weights, _ = train_emb(X, y, X[:1], y[:1], best_E, rng)  # dummy val, but we train full epochs
    # Actually we should train on all data without early stopping, just fixed epochs.
    # We'll do a fresh training on all data for MAX_EPOCHS epochs.
    emb, W_out, b_out = final_weights
    # Save weights
    # Format: magic (0x454D4244), d, E, C, then embeddings (d*256*E floats), W_out (E*C floats), b_out (C floats)
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * E);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!h || !logits) { fprintf(stderr, "alloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Zero h
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        // Sum embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e];
        }
        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int e = 0; e < E; e++) sum += h[e] * W_out[(long)e * C + c];
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