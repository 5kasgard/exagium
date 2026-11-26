```hypothesis
Propose "label-embed": a factorized linear model with per-position embeddings summed into a low-dimensional vector, followed by a linear output layer. It captures character similarities via embeddings, reducing parameters vs. label-linear (16*256*D + D*65 vs 16*256*65). Expected to beat label-linear on this character-level task because embeddings generalize across related characters. If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 30
EPOCHS_FULL = 50
CANDIDATE_D = [4, 8, 16, 32, 64]

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, D, epochs, rng):
    n, d = X.shape  # d == 16
    # Initialize parameters
    # Embeddings: (d, 256, D)
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W = rng.normal(0, math.sqrt(2.0 / (D + C)), size=(D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)
            # Forward
            # Sum embeddings: (B, D)
            summed = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                summed += E[j, xb[:, j]]
            logits = summed @ W + b  # (B, C)
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B
            # Backward
            dW = summed.T @ probs + L2 * W
            db = probs.sum(axis=0)
            dsummed = probs @ W.T
            # Gradient for embeddings
            dE = np.zeros_like(E)
            for j in range(d):
                # For each position, accumulate gradient for the used byte
                np.add.at(dE[j], xb[:, j], dsummed)
            dE += L2 * E
            # Adam update
            t += 1
            for param, grad, m, v in [
                (E, dE, mE, vE),
                (W, dW, mW, vW),
                (b, db, mb, vb)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
    return E, W, b

def evaluate(X, y, E, W, b):
    n, d = X.shape
    summed = np.zeros((n, E.shape[2]), dtype=np.float32)
    for j in range(d):
        summed += E[j, X[:, j]]
    logits = summed @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = None, -1.0
    for D in CANDIDATE_D:
        # Train on train split
        E, W, b = train_model(X_train, y_train, C, D, EPOCHS_TUNE, rng)
        acc = evaluate(X_tune, y_tune, E, W, b)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # Retrain on all data with best_D
    rng_final = np.random.default_rng(SEED)  # reset for deterministic final training
    E, W, b = train_model(X, y, C, best_D, EPOCHS_FULL, rng_final)

    # Write weights.bin
    # Magic: 0x454D4244 ('DBME' little-endian)
    magic = 0x454D4244
    out = struct.pack("<4i", magic, d, best_D, C)
    # Embeddings: (d, 256, D) -> flatten in order j, byte, dim
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < D; u++) summed[u] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) summed[u] += e[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < D; u++) {
            float su = summed[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += su * wu[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```