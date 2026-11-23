```hypothesis
Family: label-embed (position-specific byte embeddings + linear classifier).  
Why: The task is next-character prediction from 16 bytes of context. A linear model on one-hot bytes (label-linear) has 266K parameters and achieves 33.75% accuracy. By learning a small embedding (dim=8) for each of the 16 positions, we reduce parameters to ~20K, forcing the model to share statistical strength across byte values and positions, which should improve generalization on the modest 24K training set. The embedding lookup + single matrix multiply is also extremely fast at inference.  
Failure condition: If dev accuracy does not exceed 0.34 (label-linear) with comparable or lower CPU cost, the hypothesis is not supported.
```

```python
"""label-embed: position-specific byte embeddings (dim D) followed by a linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exported weights format (little-endian):
  int32 magic=0x454D4244, int32 d=16, int32 D, int32 C=65
  float32 embeddings[d][256][D]   (row-major: position, byte, dim)
  float32 W[d*D][C]               (row-major: input feature, class)
  float32 b[C]
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 8
SEED = 1
LR = 0.01
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5
L2 = 1e-5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]
    # Split: last 20% for tuning (file order)
    n_tune = n // 5
    idx_train = np.arange(n - n_tune)
    idx_tune = np.arange(n - n_tune, n)
    X_tr, y_tr = X_bytes[idx_train], y[idx_train]
    X_tu, y_tu = X_bytes[idx_tune], y[idx_tune]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, D)
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # W: (d*D, C)
    W = rng.normal(0, 0.01, size=(d * D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [E, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_acc = -1.0
    best_epoch = 0
    best_params = None
    wait = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: for each position, gather embeddings
        # Result: (B, d, D)
        emb = np.empty((B, d, D), dtype=np.float32)
        for j in range(d):
            emb[:, j, :] = E[j, X_batch[:, j]]
        # Flatten to (B, d*D)
        flat = emb.reshape(B, d * D)
        logits = flat @ W + b
        # Softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return flat, probs

    def evaluate(X, y):
        flat, probs = forward(X)
        preds = probs.argmax(axis=1)
        return (preds == y).mean()

    # Training loop on train_part with tuning validation
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            B = len(yb)

            flat, probs = forward(Xb)
            # Cross-entropy gradient
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # Gradients
            dW = flat.T @ dlogits + L2 * W
            db = dlogits.sum(axis=0)
            dflat = dlogits @ W.T  # (B, d*D)
            # Reshape to (B, d, D)
            demb = dflat.reshape(B, d, D)
            # Gradient for embeddings: scatter-add per position
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE[j], Xb[:, j], demb[:, j, :])

            grads = [dE, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Evaluate on tuning split
        acc = evaluate(X_tu, y_tu)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Retrain on full dataset for best_epoch epochs
    if best_params is not None:
        E, W, b = best_params
    params = [E, W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            B = len(yb)

            flat, probs = forward(Xb)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            dW = flat.T @ dlogits + L2 * W
            db = dlogits.sum(axis=0)
            dflat = dlogits @ W.T
            demb = dflat.reshape(B, d, D)
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE[j], Xb[:, j], demb[:, j, :])

            grads = [dE, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    E_final, W_final, b_final = params
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<4i", magic, d, D, C)
    # Embeddings: d * 256 * D floats
    emb_bytes = E_final.astype("<f4").tobytes()
    # W: (d*D) * C floats
    W_bytes = W_final.astype("<f4").tobytes()
    # b: C floats
    b_bytes = b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D4244, int32 d=16, int32 D, int32 C=65
 *   float32 embeddings[d][256][D]
 *   float32 W[d*D][C]
 *   float32 b[C]
 */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and flatten
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, D * sizeof(float));
        }

        // Linear layer: logits = flat @ W + b
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int f = 0; f < d * D; f++) {
            float val = flat[f];
            if (val == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) logits[c] += val * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```