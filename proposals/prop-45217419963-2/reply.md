```hypothesis
Family: label-embed-mlp. Uses per-position byte embeddings (E=8) followed by one ReLU hidden layer (H=64) and a linear output. Embeddings capture character similarities; hidden layer models interactions. Should beat label-linear (0.3375) on dev because tinyshakespeare benefits from sub-word composition. Failure if dev accuracy <= 0.3375 or Ir/item > 20000 (label-gbdt cost).
```
```python
"""label-embed-mlp: positional embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses E,H on last 20% of train.in, then trains on all.
Weights format: magic 0x454D4231, d, E, H, C; float32 emb[d][256][E], W1[(d*E)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-2, 1e-4, 64, 600000, 100
# Hyperparameter grid for tuning split
E_CHOICES = [4, 8, 16]
H_CHOICES = [32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_va, y_va, d, C, E, H, rng):
    # Build embeddings: one per position
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    k = d * E
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X):
        # X: (n, d) uint8
        n = X.shape[0]
        # Embedding lookup: (n, d, E)
        e = emb[X, np.arange(d)]  # advanced indexing: emb[X[i,j], j] -> (n, d, E)
        flat = e.reshape(n, -1)   # (n, d*E)
        h = np.maximum(flat @ W1 + b1, 0)
        logits = h @ W2 + b2
        return logits

    def loss_grad(X, y):
        n = X.shape[0]
        e = emb[X, np.arange(d)]
        flat = e.reshape(n, -1)
        h_pre = flat @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        # Softmax cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        # Gradients
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(0)
        gh = probs @ W2.T
        gh[h_pre <= 0] = 0
        gW1 = flat.T @ gh + L2 * W1
        gb1 = gh.sum(0)
        # Embedding gradients: gemb[d, 256, E]
        gflat = gh @ W1.T  # (n, d*E)
        gemb = np.zeros_like(emb)
        gflat_3d = gflat.reshape(n, d, E)
        # scatter-add: for each position j, for each example i, add gflat_3d[i,j] to gemb[j, X[i,j]]
        for j in range(d):
            np.add.at(gemb[j], X[:, j], gflat_3d[:, j])
        gemb += L2 * emb
        return [gemb, gW1, gb1, gW2, gb2]

    t = 0
    best_acc = -1.0
    best_params = None
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / X_tr.shape[0]))):
        order = rng.permutation(X_tr.shape[0])
        for s in range(0, X_tr.shape[0], BATCH):
            idx = order[s:s + BATCH]
            grads = loss_grad(X_tr[idx], y_tr[idx])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        logits_va = forward(X_va)
        acc = (logits_va.argmax(1) == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
    return best_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for E in E_CHOICES:
        for H in H_CHOICES:
            acc, params = train_eval(X_tr, y_tr, X_va, y_va, d, C, E, H, rng)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (E, H)
                best_params = params
    # Retrain on all data with best config
    E, H = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    _, final_params = train_eval(X, y, X[:1], y[:1], d, C, E, H, rng)  # dummy val, but train_eval returns best_params
    # Actually, we need to train on all data without early stopping. Let's do a simple full training.
    # Re-initialize with best config and train on all data for fixed epochs.
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    k = d * E
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X):
        e = emb[X, np.arange(d)]
        flat = e.reshape(X.shape[0], -1)
        h = np.maximum(flat @ W1 + b1, 0)
        return h @ W2 + b2

    def loss_grad(X, y):
        n = X.shape[0]
        e = emb[X, np.arange(d)]
        flat = e.reshape(n, -1)
        h_pre = flat @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(0)
        gh = probs @ W2.T
        gh[h_pre <= 0] = 0
        gW1 = flat.T @ gh + L2 * W1
        gb1 = gh.sum(0)
        gemb = np.zeros_like(emb)
        gflat = gh @ W1.T
        gflat_3d = gflat.reshape(n, d, E)
        for j in range(d):
            np.add.at(gemb[j], X[:, j], gflat_3d[:, j])
        gemb += L2 * emb
        return [gemb, gW1, gb1, gW2, gb2]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            grads = loss_grad(X[idx], y[idx])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    emb, W1, b1, W2, b2 = params
    # Pack weights
    out = struct.pack("<5i", 0x454D4231, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input: 16 bytes. Output: 1 byte class index.
 * Weights: magic 0x454D4231, d, E, H, C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and flatten
        float *flat = hidden; // reuse hidden as flat buffer of size d*E (H >= d*E? not necessarily, allocate separately)
        // Actually we need a buffer of size d*E. Let's allocate flat separately.
    }
    // Rework with proper buffers
    free(hidden); free(logits);
    float *flat = malloc(sizeof(float) * d * E);
    hidden = malloc(sizeof(float) * H);
    logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, sizeof(float) * E);
        }
        // Hidden layer: flat (d*E) -> H
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h * d * E; // Wait: W1 is [d*E][H] row-major? In train.py: W1 shape (k, H) where k=d*E. So W1[flat_idx, h].
            // In train.py: flat @ W1 -> (n, H). So W1 is (k, H) column-major? Actually numpy: (k, H) @ (H,) -> (k,)? Let's check.
            // flat shape (n, k), W1 shape (k, H). So flat @ W1 -> (n, H). So W1 is stored row-major: W1[flat_idx * H + h]? No.
            // In numpy, (k, H) array is row-major: row stride H, column stride 1. So element (i, j) at [i*H + j].
            // But we did W1 = (rng... (k, H)). So W1[i, h] is at index i*H + h.
            // In inference, we have flat[i] for i in 0..k-1. So sum += flat[i] * W1[i*H + h].
            // Let's compute correctly.
            for (int idx = 0; idx < d * E; idx++) {
                sum += flat[idx] * W1[idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer: hidden (H) -> C
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c]; // W2 shape (H, C) row-major: W2[h, c] at h*C + c.
            }
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