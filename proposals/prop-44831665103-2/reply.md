```hypothesis
Family: label-emb-mlp — per-position byte embeddings (E=8) concatenated, one ReLU hidden layer (H=64), then linear to 65 classes.
Why: The task is character-level language modeling with 16-byte context. One-hot linear regression (label-linear) ignores positional interactions and has 266k parameters. Learned embeddings capture byte similarities and position-specific patterns with far fewer parameters (~45k), and a single hidden layer adds non-linearity to model n-gram interactions. This should improve accuracy over label-linear while keeping inference fast (small dense matrix multiplies).
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds ~15k Ir/item, the family does not justify its complexity.
```

```python
"""label-emb-mlp: per-position embeddings + one ReLU hidden layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Hyperparameters (E=8, H=64) chosen on the last 20% tuning split of train.in, then retrained on all data.
Exported weights format (little-endian):
  int32 magic=0x454D4231 ('EMB1'), d, E, H, C
  float32 emb[d][256][E]       (row-major: position, byte, embedding dim)
  float32 W1[d*E][H]           (row-major)
  float32 b1[H]
  float32 W2[H][C]             (row-major)
  float32 b2[C]
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned on tuning split)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, epochs, rng, emb, W1, b1, W2, b2):
    """Train on given data for given number of epochs, updating parameters in-place."""
    n = X.shape[0]
    # Adam state
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsize = len(yb)
            # Forward
            # Embedding lookup: (b, d, E)
            emb_vals = np.stack([emb[j, xb[:, j]] for j in range(d)], axis=1)  # (b, d, E)
            z = emb_vals.reshape(bsize, d * E)                                 # (b, d*E)
            h_pre = z @ W1 + b1                                                # (b, H)
            h = np.maximum(h_pre, 0)                                           # ReLU
            logits = h @ W2 + b2                                               # (b, C)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            grad_logits = probs / bsize                                        # (b, C)
            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2                              # (H, C)
            grad_b2 = grad_logits.sum(axis=0)                                  # (C,)
            grad_h = grad_logits @ W2.T                                        # (b, H)
            grad_h_pre = grad_h * (h_pre > 0)                                  # ReLU derivative
            grad_W1 = z.T @ grad_h_pre + L2 * W1                               # (d*E, H)
            grad_b1 = grad_h_pre.sum(axis=0)                                   # (H,)
            grad_z = grad_h_pre @ W1.T                                         # (b, d*E)
            grad_emb_vals = grad_z.reshape(bsize, d, E)                        # (b, d, E)
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_vals[:, j])
            grad_emb += L2 * emb
            # Adam update
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def evaluate(X, y, d, C, emb, W1, b1, W2, b2):
    """Return accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        bsize = len(yb)
        emb_vals = np.stack([emb[j, xb[:, j]] for j in range(d)], axis=1)
        z = emb_vals.reshape(bsize, d * E)
        h = np.maximum(z @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # Split: last 20% for tuning (validation), first 80% for training during search
    split = int(0.8 * n_all)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    # Simple search over H on validation set (E fixed)
    best_H = H
    best_acc = -1.0
    for cand_H in [32, 64, 128]:
        # Initialize parameters
        emb = (rng.normal(0, 1, (d, 256, E)) * math.sqrt(2.0 / (256 + E))).astype(np.float32)
        W1 = (rng.normal(0, 1, (d * E, cand_H)) * math.sqrt(2.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(cand_H, np.float32)
        W2 = (rng.normal(0, 1, (cand_H, C)) * math.sqrt(2.0 / cand_H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        # Quick training (5 epochs) on training split
        train_model(X_train, y_train, d, C, epochs=5, rng=rng, emb=emb, W1=W1, b1=b1, W2=W2, b2=b2)
        acc = evaluate(X_val, y_val, d, C, emb, W1, b1, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_H = cand_H
    # Retrain on all data with best_H
    H_final = best_H
    emb = (rng.normal(0, 1, (d, 256, E)) * math.sqrt(2.0 / (256 + E))).astype(np.float32)
    W1 = (rng.normal(0, 1, (d * E, H_final)) * math.sqrt(2.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H_final, np.float32)
    W2 = (rng.normal(0, 1, (H_final, C)) * math.sqrt(2.0 / H_final)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    train_model(X_all, y_all, d, C, epochs=epochs, rng=rng, emb=emb, W1=W1, b1=b1, W2=W2, b2=b2)
    # Pack weights
    # magic, d, E, H, C
    header = struct.pack("<5i", 0x454D4231, d, E, H_final, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W1: (d*E) * H_final
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
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D4231, d, E, H, C
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Initialize hidden pre-activation with bias
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        // Accumulate embedding contributions
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) {
                float ev = e[ee];
                const float *w1_row = W1 + (j * E + ee) * H;
                for (int h = 0; h < H; h++) hidden[h] += ev * w1_row[h];
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        // Output scores
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            const float *w2_row = W2 + h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2_row[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(hidden);
    free(scores);
    return 0;
}
```