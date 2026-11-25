```hypothesis
Family: label-embed (per-position embeddings + linear classifier). 
Why: The 16-byte context consists of categorical bytes; one-hot linear model (label-linear) has 4096×65 weights but cannot share statistical strength across similar characters. Embeddings (16×256×E) allow generalization across the alphabet and reduce parameters. A linear layer on top (16·E → 65) keeps inference fast. 
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%, the hypothesis is wrong.
```

```python
import json
import math
import struct
import sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 64                 # embedding dimension
SEED = 1
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
L2 = 1e-5
BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 5
VAL_FRAC = 0.2         # last 20% for validation

MAGIC = 0x454D4245     # "EMBE"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy(logits, y):
    # logits: (B, C), y: (B,) int
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    probs[np.arange(len(y)), y] -= 1.0
    grad = probs / len(y)  # (B, C)
    return loss, grad

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65

    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Split: last 20% validation (file order)
    val_start = int(n * (1 - VAL_FRAC))
    idx_train = np.arange(val_start)
    idx_val = np.arange(val_start, n)
    X_train, y_train = X_bytes[idx_train], y[idx_train]
    X_val, y_val = X_bytes[idx_val], y[idx_val]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(256), (d, 256, E))).astype(np.float32)
    # Initialize linear: (d*E, C)
    W = (rng.normal(0, 1.0 / math.sqrt(d * E), (d * E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    best_val_loss = float('inf')
    best_params = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Shuffle training data
        order = rng.permutation(len(X_train))
        X_train_shuf = X_train[order]
        y_train_shuf = y_train[order]

        # Training batches
        for s in range(0, len(X_train), BATCH):
            xb = X_train_shuf[s:s+BATCH]
            yb = y_train_shuf[s:s+BATCH]
            bsz = len(xb)

            # Forward: embeddings lookup
            # xb: (bsz, d) -> for each position, gather emb[pos, xb[:,pos], :] -> (bsz, d, E)
            emb_seq = np.stack([emb[pos, xb[:, pos], :] for pos in range(d)], axis=1)  # (bsz, d, E)
            flat = emb_seq.reshape(bsz, d * E)  # (bsz, d*E)
            logits = flat @ W + b  # (bsz, C)

            # Loss and gradient on logits
            loss, grad_logits = softmax_cross_entropy(logits, yb)

            # Backprop
            grad_W = flat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ W.T  # (bsz, d*E)
            grad_emb_seq = grad_flat.reshape(bsz, d, E)  # (bsz, d, E)

            # Accumulate gradients for embeddings
            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                # grad_emb_seq[:, pos, :] -> (bsz, E)
                # indices: xb[:, pos]
                np.add.at(grad_emb[pos], xb[:, pos], grad_emb_seq[:, pos, :])

            grads = [grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * (g * g)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

        # Validation
        emb_seq_val = np.stack([emb[pos, X_val[:, pos], :] for pos in range(d)], axis=1)
        flat_val = emb_seq_val.reshape(len(X_val), d * E)
        logits_val = flat_val @ W + b
        val_loss, _ = softmax_cross_entropy(logits_val, y_val)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_params = [p.copy() for p in params]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Use best params
    if best_params is not None:
        emb, W, b = best_params

    # Retrain on full data? The contract says: choose settings on tuning split, then train on all of train.
    # We already used the tuning split for early stopping. Now we do a few epochs on full data with the same hyperparams.
    # But we can also just keep the best model from the split (which was trained on 80%). To follow contract, we retrain on all data.
    # We'll do a few more epochs on full data starting from best_params.
    X_full, y_full = X_bytes, y
    for epoch in range(10):  # few epochs on full data
        order = rng.permutation(n)
        X_full_shuf = X_full[order]
        y_full_shuf = y_full[order]
        for s in range(0, n, BATCH):
            xb = X_full_shuf[s:s+BATCH]
            yb = y_full_shuf[s:s+BATCH]
            bsz = len(xb)
            emb_seq = np.stack([emb[pos, xb[:, pos], :] for pos in range(d)], axis=1)
            flat = emb_seq.reshape(bsz, d * E)
            logits = flat @ W + b
            loss, grad_logits = softmax_cross_entropy(logits, yb)
            grad_W = flat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ W.T
            grad_emb_seq = grad_flat.reshape(bsz, d, E)
            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                np.add.at(grad_emb[pos], xb[:, pos], grad_emb_seq[:, pos, :])
            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * (g * g)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Save weights
    # Format: magic (i32), d (i32), E (i32), C (i32)
    # then embeddings: d * 256 * E float32 (row-major: pos, byte, feature)
    # then W: (d*E) * C float32
    # then b: C float32
    header = struct.pack("<4i", MAGIC, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + emb_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (0..C-1).
 * Weights format: magic(0x454D4245), d, E, C (int32 each), then embeddings[d][256][E] (float32),
 * W[(d*E)][C] (float32), b[C] (float32). All little-endian. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * E;
            memcpy(flat + p * E, e, E * sizeof(float));
        }

        // Linear layer: logits = flat @ W + b
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int j = 0; j < d * E; j++) {
            float f = flat[j];
            if (f == 0.0f) continue;
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) logits[c] += f * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(flat);
    free(logits);
    return 0;
}
```