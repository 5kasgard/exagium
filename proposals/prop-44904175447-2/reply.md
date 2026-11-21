```hypothesis
Family: label-embed (positional embeddings + linear classifier).
Why: The task is next-character prediction from 16 bytes of context. Each byte is a categorical token; a linear model on one-hot features (label-linear) cannot capture interactions between positions or semantic similarity between byte values. Learned positional embeddings map each byte at each position into a low-dimensional space where similar characters (e.g., letters) are close, allowing the model to generalize across contexts. Summing embeddings is parameter-efficient (16*256*D + D*65) and fast at inference (O(16*D + D*65) operations). With D=32 this is ~130k parameters, well within memory limits, and should outperform label-linear (0.3375) while staying far cheaper than label-gbdt (131k Ir/item).
Failure signal: If dev accuracy ≤ 0.34 or Ir/item > 20000, the model does not justify its complexity over label-linear.
```

```python
"""label-embed: positional embeddings + linear classifier for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameters (embedding dim D, lr, epochs) chosen on last 20% of train.in (tuning split),
then final model trained on all train.in.
Weights format (little-endian):
  int32 magic=0x454D4245 ('EMBE'), int32 d=16, int32 D, int32 C=65
  float32 embeddings[d][256][D]   (flattened: d*256*D)
  float32 W_out[D][C]             (flattened: D*C)
  float32 b_out[C]                (flattened: C)
"""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRAC = 0.2
# Hyperparameter grid for tuning
D_CHOICES = [16, 32, 64]
LR_CHOICES = [0.01, 0.03]
EPOCHS_PER_TUNE = 3
FINAL_EPOCHS = 10
BATCH = 256
L2 = 1e-5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (already averaged over batch)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_on_split(X, y, D, lr, epochs, rng):
    n, d = X.shape
    C = 65
    # Initialize embeddings: d * 256 * D
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)
            # Forward: sum embeddings
            h = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]   # (B, D)
            logits = h @ W + b          # (B, C)
            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)   # (B, C)
            # Gradients
            g_h = g_logits @ W.T        # (B, D)
            g_W = h.T @ g_logits + L2 * W   # (D, C)
            g_b = g_logits.sum(axis=0)       # (C,)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # accumulate gradient for each position
                np.add.at(g_emb[j], xb[:, j], g_h)
            g_emb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, g_emb, m_emb, v_emb),
                (W, g_W, m_W, v_W),
                (b, g_b, m_b, v_b)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return emb, W, b

def evaluate(emb, W, b, X, y):
    n, d = X.shape
    h = np.zeros((n, W.shape[0]), dtype=np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    logits = h @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    split = int(n * (1 - TUNE_FRAC))
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_model = None
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            rng_tune = np.random.default_rng(SEED)
            emb, W, b = train_on_split(X_train, y_train, D, lr, EPOCHS_PER_TUNE, rng_tune)
            acc = evaluate(emb, W, b, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (D, lr)
                best_model = (emb, W, b)
    print(f"Tuning: best D={best_params[0]}, lr={best_params[1]}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    D_best, lr_best = best_params
    rng_full = np.random.default_rng(SEED)
    emb, W, b = train_on_split(X_all, y_all, D_best, lr_best, FINAL_EPOCHS, rng_full)

    # Write weights
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<4i", magic, d, D_best, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D4245, int32 d=16, int32 D, int32 C=65
 *   float32 emb[d][256][D]   (d*256*D floats)
 *   float32 W_out[D][C]      (D*C floats)
 *   float32 b_out[C]         (C floats)
 */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) h[v] += e[v];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int v = 0; v < D; v++) {
            float hv = h[v];
            if (hv == 0.0f) continue;
            const float *wv = W + (long)v * C;
            for (int c = 0; c < C; c++) scores[c] += hv * wv[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```