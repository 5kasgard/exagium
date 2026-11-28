```hypothesis
Family: emb-linear (per-position byte embeddings + linear classifier).
Why: The input is 16 categorical bytes; label-linear uses a 4096-d one-hot vector. Replacing one-hot with learned 8-d embeddings per position reduces parameters 32× (4096→128), cuts inference FLOPs by ~30×, and acts as a regularizer that should improve generalization on the small tinyshakespeare training set (24k examples). The model is a simple two-layer network (embedding lookup + linear) with no hidden non-linearity, so inference is a few thousand FLOPs per item.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or instructions/item ≥ 7359.9 (label-linear), the hypothesis fails.
```

```python
"""emb-linear: per-position byte embeddings (D=8) + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
D = 8                 # embedding dimension per position
LR = 1e-2
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
VAL_FRAC = 0.2        # last 20% for validation (card's tuning split)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]          # d=16, C=65
    n = X.shape[0]
    val_n = int(n * VAL_FRAC)
    train_n = n - val_n

    rng = np.random.default_rng(SEED)

    # Embeddings: d * 256 * D
    E = (rng.normal(0, 1.0 / math.sqrt(D), (d, 256, D))).astype(np.float32)
    # Linear weights: (d*D) x C
    W = (rng.normal(0, 1.0 / math.sqrt(d * D), (d * D, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [E.reshape(-1), W.reshape(-1), b]
    shapes = [E.shape, W.shape, b.shape]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def pack_params(E_flat, W_flat, b_vec):
        E_mat = E_flat.reshape(d, 256, D)
        W_mat = W_flat.reshape(d * D, C)
        return E_mat, W_mat, b_vec

    def forward(E_mat, W_mat, b_vec, X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, D)
        emb = E_mat[np.arange(d), X_batch.T].transpose(1, 0, 2)  # (B, d, D)
        # Flatten to (B, d*D)
        flat = emb.reshape(B, d * D)
        # Linear
        logits = flat @ W_mat + b_vec
        return logits, flat

    def loss_grad(logits, y_batch, flat):
        B = logits.shape[0]
        # Softmax + cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B
        # Gradient w.r.t. linear weights and bias
        grad_W = flat.T @ probs + L2 * W_mat
        grad_b = probs.sum(axis=0)
        # Gradient w.r.t. embeddings
        grad_flat = probs @ W_mat.T
        grad_emb = grad_flat.reshape(B, d, D)
        # Scatter-add to embedding gradient
        grad_E = np.zeros_like(E_mat, dtype=np.float32)
        np.add.at(grad_E, (np.arange(d), X_batch.T), grad_emb.transpose(1, 0, 2))
        grad_E += L2 * E_mat
        return grad_E.reshape(-1), grad_W.reshape(-1), grad_b

    def evaluate(E_mat, W_mat, b_vec, X_eval, y_eval):
        logits, _ = forward(E_mat, W_mat, b_vec, X_eval)
        preds = logits.argmax(axis=1)
        return (preds == y_eval).mean()

    # Split
    X_train, y_train = X[:train_n], y[:train_n]
    X_val, y_val = X[train_n:], y[train_n:]

    best_val_acc = -1.0
    best_params = None
    t = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(train_n)
        for s in range(0, train_n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            E_mat, W_mat, b_vec = pack_params(params[0], params[1], params[2])
            logits, flat = forward(E_mat, W_mat, b_vec, X_batch)
            grads = loss_grad(logits, y_batch, flat)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation
        E_mat, W_mat, b_vec = pack_params(params[0], params[1], params[2])
        val_acc = evaluate(E_mat, W_mat, b_vec, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on full data for the same number of epochs that gave best val? Simpler: use best_params and do a few more epochs on full data.
    # We'll just use the best_params and fine-tune on full data for 5 epochs.
    params = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(5):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            E_mat, W_mat, b_vec = pack_params(params[0], params[1], params[2])
            logits, flat = forward(E_mat, W_mat, b_vec, X_batch)
            grads = loss_grad(logits, y_batch, flat)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Final packing
    E_mat, W_mat, b_vec = pack_params(params[0], params[1], params[2])

    # Write weights.bin
    # Header: magic, d, D, C (int32 little-endian)
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<4i", magic, d, D, C)
    # Embeddings: d * 256 * D floats (float32 little-endian)
    emb_bytes = E_mat.astype("<f4").tobytes()
    # Linear weights: (d*D) * C floats
    w_bytes = W_mat.astype("<f4").tobytes()
    # Bias: C floats
    b_bytes = b_vec.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + emb_bytes + w_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* emb-linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: int32 magic(0x454D4244), d, D, C; float32 E[d][256][D], W[d*D][C], b[C]. */
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
    if (d != 16 || D != 8 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize with bias
        memcpy(scores, b, C * sizeof(float));

        // Embedding lookup and accumulate into emb_sum
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            float *dest = emb_sum + j * D;
            for (int kk = 0; kk < D; kk++) dest[kk] = e[kk];
        }

        // Linear layer: scores += emb_sum @ W
        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int kk = 0; kk < d * D; kk++) acc += emb_sum[kk] * W[(long)kk * C + c];
            scores[c] += acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```