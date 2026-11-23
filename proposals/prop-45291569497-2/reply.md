```hypothesis
Family: label-emb (per-position embeddings + linear). The 16-byte context is categorical; a linear model uses 16 independent 256×65 weight tables (266k params). Factorizing each table into a 256×E embedding and a shared E×65 projection (E≈32) cuts parameters to ~164k, reduces multiply-adds by 8×, and lets the model generalize across similar characters (e.g., letters). This should beat label-linear (0.3375) on accuracy while being faster. If it fails, the task either has no character-level similarity or the bottleneck loses critical positional information.
```
```python
"""label-emb: per-position embeddings + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy+stdlib only.
Tunes embedding dim and learning rate on last 20% of train.in (by file order),
then retrains on all train.in. Exports weights as:
  int32 magic=0x4C424531 ('EBL1'), d, E, C
  float32 emb[d][256][E]   (row-major per position)
  float32 W[d*E][C]
  float32 b[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
# Tuning grid
EMB_DIMS = [16, 32, 48, 64]
LRS = [1e-3, 3e-3, 1e-2]
TUNE_EPOCHS = 15
FINAL_EPOCHS = 50
BATCH = 64
L2 = 1e-4
SAMPLES_PER_EPOCH = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """logits: (B, C), y: (B,) int. Returns gradient wrt logits (B, C) and loss scalar."""
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    probs /= B
    loss = -np.log(np.maximum(probs[np.arange(B), y] + 1.0, 1e-12)).mean()  # actually probs after subtraction? Let's recompute correctly.
    # Correct loss:
    log_probs = logits - np.log(exp_logits.sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(B), y].mean()
    return probs.astype(np.float32), float(loss)


def evaluate(X, y, emb, W, b):
    n = X.shape[0]
    d, E = emb.shape[0], emb.shape[2]
    C = W.shape[1]
    correct = 0
    for i in range(n):
        # concatenate embeddings
        vec = np.concatenate([emb[j, X[i, j]] for j in range(d)])
        scores = vec @ W + b
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n


def train_model(X, y, d, C, E, lr, epochs, rng):
    # Initialize
    emb = rng.uniform(-0.1, 0.1, (d, 256, E)).astype(np.float32)
    W = rng.uniform(-1, 1, (d * E, C)).astype(np.float32) * math.sqrt(6.0 / (d * E))
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    steps_per_epoch = max(1, math.ceil(SAMPLES_PER_EPOCH / n))
    total_steps = epochs * steps_per_epoch

    for epoch in range(epochs):
        order = rng.permutation(n)
        for step in range(steps_per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            if len(idx) == 0:
                break
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward: gather embeddings
            # emb shape (d, 256, E) -> for each position j, we need emb[j, xb[:, j]] -> (B, E)
            # concatenate along axis=1 -> (B, d*E)
            vecs = np.empty((B, d * E), np.float32)
            for j in range(d):
                vecs[:, j * E:(j + 1) * E] = emb[j, xb[:, j]]

            logits = vecs @ W + b  # (B, C)

            # Loss and gradient
            grad_logits, loss = softmax_cross_entropy_grad(logits, yb)

            # Gradient wrt W, b
            grad_W = vecs.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)

            # Gradient wrt embeddings
            grad_emb = np.zeros_like(emb)
            grad_vecs = grad_logits @ W.T  # (B, d*E)
            for j in range(d):
                g = grad_vecs[:, j * E:(j + 1) * E]  # (B, E)
                # scatter add: for each sample i, add g[i] to emb[j, xb[i, j]]
                np.add.at(grad_emb[j], xb[:, j], g)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat = m_b / (1 - beta1 ** t)
            v_hat = v_b / (1 - beta2 ** t)
            b -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning (by file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_cfg = None
    best_weights = None

    print("Tuning on validation set...", file=sys.stderr)
    for E in EMB_DIMS:
        for lr in LRS:
            rng_tune = np.random.default_rng(SEED)  # fresh RNG for each config
            emb, W, b = train_model(X_train, y_train, d, C, E, lr, TUNE_EPOCHS, rng_tune)
            acc = evaluate(X_val, y_val, emb, W, b)
            print(f"  E={E}, lr={lr:.0e} -> val acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (E, lr)
                best_weights = (emb.copy(), W.copy(), b.copy())

    E, lr = best_cfg
    print(f"Best config: E={E}, lr={lr:.0e} (val acc={best_acc:.4f})", file=sys.stderr)

    # Retrain on full data
    rng_final = np.random.default_rng(SEED)
    emb, W, b = train_model(X, y, d, C, E, lr, FINAL_EPOCHS, rng_final)

    # Write weights.bin
    # Magic, d, E, C
    header = struct.pack("<4i", 0x4C424531, d, E, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W: (d*E) * C floats
    W_bytes = W.astype("<f4").tobytes()
    # b: C floats
    b_bytes = b.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(emb_bytes)
        f.write(W_bytes)
        f.write(b_bytes)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x4C424531 ('EBL1')
 *   int32 d, E, C
 *   float32 emb[d][256][E]   (row-major: position, byte, embedding)
 *   float32 W[d*E][C]
 *   float32 b[C] */
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
    if (nw < 16 || hd[0] != 0x4C424531) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(vec + j * E, e, E * sizeof(float));
        }

        // Linear layer: scores = vec @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * E; idx++) {
            float v = vec[idx];
            if (v != 0.0f) {
                const float *w = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```