```hypothesis
Family: label-embed (shared byte embeddings + linear classifier). The 16-byte context is mapped through a shared 256×E embedding table (E=32), concatenated, and fed to a linear layer (512→65). This reduces parameters vs. label-linear (41k vs 266k) while preserving positional information, and should capture character similarities. If dev accuracy ≤ 0.3375 or inference cost ≥ label-linear, the hypothesis fails.
```
```python
"""label-embed: shared byte embeddings + linear classifier for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
EMB_DIMS = [16, 32, 64]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
VAL_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_val, y_val, emb_dim, C, rng):
    d = X_tr.shape[1]
    # init embeddings and linear weights
    E = (rng.uniform(-1, 1, (256, emb_dim)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d * emb_dim, C)) * math.sqrt(6.0 / (d * emb_dim))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_tr = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], yb = y_tr[idx]
            # forward
            # embed: (B, d) -> (B, d, emb_dim) -> (B, d*emb_dim)
            emb = E[xb]  # (B, d, emb_dim)
            flat = emb.reshape(len(idx), -1)  # (B, d*emb_dim)
            logits = flat @ W + b
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # gradients
            grad_flat = grad_logits @ W.T  # (B, d*emb_dim)
            grad_W = flat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            # reshape grad_flat to (B, d, emb_dim) for embedding gradient
            grad_emb = np.zeros_like(E)
            np.add.at(grad_emb, xb.reshape(-1), grad_flat.reshape(-1, emb_dim))
            grad_emb += L2 * E
            grads = [grad_emb, grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    emb_val = E[X_val].reshape(X_val.shape[0], -1)
    logits_val = emb_val @ W + b
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    # split: last 20% for validation
    n = X.shape[0]
    split = int(n * (1 - VAL_FRAC))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    best_acc, best_params, best_dim = -1, None, EMB_DIMS[0]
    for emb_dim in EMB_DIMS:
        acc, params = train_eval(X_tr, y_tr, X_val, y_val, emb_dim, C, rng)
        if acc > best_acc:
            best_acc, best_params, best_dim = acc, params, emb_dim
    # retrain on all data with best_dim
    rng = np.random.default_rng(SEED)  # reset for reproducibility
    _, final_params = train_eval(X, y, X[:1], y[:1], best_dim, C, rng)  # dummy val, but train_eval trains on X_tr
    # Actually train_eval returns params trained on X_tr; we need a function that trains on given data.
    # Let's just call train_eval with X as both train and val (val tiny) to get final params.
    # But train_eval uses X_tr for training and X_val for validation only. We'll create a final_train function.
    # For simplicity, we'll reuse train_eval with X_val = X[:1] (won't affect training).
    # However, train_eval trains for epochs based on SAMPLES / n_tr. With n_tr = n, it's fine.
    # We already have final_params from the call above? No, we called train_eval with X_tr=X, y_tr=y, X_val=X[:1], y_val=y[:1].
    # That trains on all data. Good.
    E_final, W_final, b_final = final_params
    # Write weights.bin
    # Magic: 0x454D424C ('EMBL' little-endian)
    magic = 0x454D424C
    out = struct.pack("<4i", magic, d, best_dim, C)
    out += E_final.astype("<f4").tobytes()
    out += W_final.astype("<f4").tobytes()
    out += b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output: one byte class index (0..C-1).
 * Weights format: magic(0x454D424C), int32 d, emb_dim, C; float32 E[256][emb_dim], W[d*emb_dim][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)256 * emb_dim + (long)d * emb_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)256 * emb_dim;
    const float *b = W + (long)d * emb_dim * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // init logits with bias
        for (int c = 0; c < C; c++) logits[c] = b[c];
        // accumulate W^T * embeddings
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * emb_dim;
            const float *w = W + (long)j * emb_dim * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int k = 0; k < emb_dim; k++) sum += e[k] * w[(long)k * C + c];
                logits[c] += sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```