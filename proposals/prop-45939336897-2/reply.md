```hypothesis
Family: label-emblinear. Learns a per-position byte embedding (dim E) followed by a linear layer.
Why: The 16-byte context benefits from shared low-dimensional representations, reducing parameters from 266k (one-hot linear) to ~41k (E=8) while capturing similarities between bytes. This should improve generalization and accuracy over label-linear, and be faster due to smaller matrix multiplies.
Failure: If dev accuracy <= label-linear (0.3375) or instructions per item >= label-linear (7359.9), the family does not meet its goals.
```

```python
"""label-emblinear: per-position byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all data."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
EMBED_DIMS = [4, 8, 16]
LRS = [1e-3, 3e-3]
L2S = [1e-4, 1e-5]
BATCH = 64
VAL_EPOCHS = 30
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for validation
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    best_acc = -1.0
    best_cfg = None

    rng = np.random.default_rng(SEED)

    # Grid search on validation split
    for E in EMBED_DIMS:
        for lr in LRS:
            for l2 in L2S:
                # Initialize parameters
                embeddings = rng.uniform(-1, 1, (d, 256, E)).astype(np.float32) * math.sqrt(6.0 / 256)
                W = rng.uniform(-1, 1, (d * E, C)).astype(np.float32) * math.sqrt(6.0 / (d * E))
                b = np.zeros(C, np.float32)
                params = [embeddings, W, b]
                state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
                t = 0

                n_train = X_train.shape[0]
                for _ in range(VAL_EPOCHS):
                    order = rng.permutation(n_train)
                    for s in range(0, n_train, BATCH):
                        idx = order[s:s + BATCH]
                        xb = X_train[idx]
                        yb = y_train[idx]

                        # Forward: concatenate embeddings
                        batch_emb = np.concatenate([embeddings[j, xb[:, j]] for j in range(d)], axis=1)
                        logits = batch_emb @ W + b

                        # Softmax cross-entropy gradient
                        logits = logits - logits.max(axis=1, keepdims=True)
                        exp_logits = np.exp(logits)
                        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                        probs[np.arange(len(yb)), yb] -= 1.0
                        grad_logits = probs / len(yb)  # (batch, C)

                        # Gradients
                        grad_W = batch_emb.T @ grad_logits + l2 * W
                        grad_b = grad_logits.sum(axis=0)
                        grad_emb = np.zeros_like(embeddings)
                        grad_batch_emb = grad_logits @ W.T
                        for j in range(d):
                            g = grad_batch_emb[:, j * E:(j + 1) * E]
                            np.add.at(grad_emb[j], xb[:, j], g)
                        grad_emb += l2 * embeddings

                        grads = [grad_emb, grad_W, grad_b]
                        t += 1
                        for i, (p, gr) in enumerate(zip(params, grads)):
                            m, v = state[i]
                            m *= 0.9; m += 0.1 * gr
                            v *= 0.999; v += 0.001 * gr * gr
                            m_hat = m / (1 - 0.9 ** t)
                            v_hat = v / (1 - 0.999 ** t)
                            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

                # Validate
                batch_emb_val = np.concatenate([embeddings[j, X_val[:, j]] for j in range(d)], axis=1)
                logits_val = batch_emb_val @ W + b
                acc = (logits_val.argmax(axis=1) == y_val).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (E, lr, l2)

    # Retrain on all data with best config
    E, lr, l2 = best_cfg
    embeddings = rng.uniform(-1, 1, (d, 256, E)).astype(np.float32) * math.sqrt(6.0 / 256)
    W = rng.uniform(-1, 1, (d * E, C)).astype(np.float32) * math.sqrt(6.0 / (d * E))
    b = np.zeros(C, np.float32)
    params = [embeddings, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(100, math.ceil(600000 / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]

            batch_emb = np.concatenate([embeddings[j, xb[:, j]] for j in range(d)], axis=1)
            logits = batch_emb @ W + b

            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)

            grad_W = batch_emb.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb = np.zeros_like(embeddings)
            grad_batch_emb = grad_logits @ W.T
            for j in range(d):
                g = grad_batch_emb[:, j * E:(j + 1) * E]
                np.add.at(grad_emb[j], xb[:, j], g)
            grad_emb += l2 * embeddings

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic(0x454D424C), d, E, C (int32), then embeddings[d][256][E], W[d*E][C], b[C] (all float32)
    magic = 0x454D424C
    out = struct.pack("<4i", magic, d, E, C)
    out += embeddings.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emblinear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, d, E, C (int32), then embeddings[d][256][E], W[d*E][C], b[C] (float32). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < d * E; k++) {
                sum += concat[k] * W[(long)k * C + c];
            }
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```