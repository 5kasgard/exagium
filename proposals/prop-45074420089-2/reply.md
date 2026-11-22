```hypothesis
Family: label-embed (factorized linear model with per-position byte embeddings).
Why: The input is 16 categorical bytes; a linear model on one-hot features (label-linear) ignores positional structure and has many parameters (4096×65). Learning a small embedding per position (16×256×E) followed by a linear layer (E×65) captures positional interactions, reduces parameters, and often generalizes better on sequence tasks. With E=32 it has ~133k parameters vs 266k for label-linear, and inference is just 16 vector adds + one matrix-vector multiply.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >2×, the factorization does not help for this task.
```

```python
"""label-embed: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then trained on all train.
Weights format: magic 0x454D4244, d=16, E, C; float32 embeddings[d*256*E], W_out[E*C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning (last 20% of train.in)
EMBED_DIMS = [16, 32, 64]
LEARNING_RATES = [0.01, 0.03, 0.1]
EPOCHS_CHOICES = [10, 20, 30]
BATCH_SIZE = 64
L2 = 1e-5
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
VAL_FRACTION = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_model(X, y, E, lr, epochs, rng, X_val=None, y_val=None):
    n, d = X.shape
    C = 65
    # Initialize parameters
    embed = (rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E))).astype(np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / (E + C)), size=(E, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)
    params = [embed, W_out, b_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            bs = len(yb)
            # Forward
            emb_sum = np.zeros((bs, E), dtype=np.float32)
            for j in range(d):
                emb_sum += embed[j, xb[:, j]]
            logits = emb_sum @ W_out + b_out
            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # Backward
            grad_W_out = emb_sum.T @ probs + L2 * W_out
            grad_b_out = probs.sum(axis=0)
            grad_emb_sum = probs @ W_out.T
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                # scatter-add grad_emb_sum into grad_embed[j, xb[:, j], :]
                np.add.at(grad_embed[j], xb[:, j], grad_emb_sum)
            grad_embed += L2 * embed
            grads = [grad_embed, grad_W_out, grad_b_out]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * gr
                v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * (gr * gr)
                m_hat = m[i] / (1 - ADAM_BETA1 ** t)
                v_hat = v[i] / (1 - ADAM_BETA2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        # Validation check
        if X_val is not None:
            emb_sum_val = np.zeros((len(y_val), E), dtype=np.float32)
            for j in range(d):
                emb_sum_val += embed[j, X_val[:, j]]
            logits_val = emb_sum_val @ W_out + b_out
            val_acc = accuracy(logits_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]
    if best_params is not None:
        params = best_params
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y)
    # Split last 20% for validation (file order)
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_hparams = None
    best_params = None
    # Grid search
    for E in EMBED_DIMS:
        for lr in LEARNING_RATES:
            for epochs in EPOCHS_CHOICES:
                params = train_model(X_train, y_train, E, lr, epochs, rng, X_val, y_val)
                embed, W_out, b_out = params
                emb_sum_val = np.zeros((val_size, E), dtype=np.float32)
                for j in range(d):
                    emb_sum_val += embed[j, X_val[:, j]]
                logits_val = emb_sum_val @ W_out + b_out
                val_acc = accuracy(logits_val, y_val)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_hparams = (E, lr, epochs)
                    best_params = params
    print(f"Best hparams: E={best_hparams[0]}, lr={best_hparams[1]}, epochs={best_hparams[2]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full training set with best hparams
    E, lr, epochs = best_hparams
    rng_full = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_model(X, y, E, lr, epochs, rng_full)
    embed, W_out, b_out = final_params
    # Write weights.bin
    magic = 0x454D4244  # 'EMBD' in little-endian uint32
    out = struct.pack("<4I", magic, d, E, C)
    out += embed.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16); output is one byte (class index).
 * Weights format: uint32 magic=0x454D4244, d, E, C; float32 embed[d*256*E], W_out[E*C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *embed = (const float *)(hd + 4);
    const float *W_out = embed + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = embed + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += eptr[e];
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int e = 0; e < E; e++) sum += acc[e] * W_out[(long)e * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(logits);
    return 0;
}
```