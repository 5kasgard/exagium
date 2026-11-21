```hypothesis
We propose "label-embed": per-position byte embeddings (16 positions × 256 values × D) followed by a linear layer to 65 classes. This compresses the 4096‑dim one‑hot input (label-linear) to 16·D (e.g., 128–512), letting the model learn character similarities and generalize better from 24k examples. Hyperparameters (D, learning rate) are chosen on the last 20% of train.in, then the model is retrained on all data. Expected to beat label-linear (0.3375) on dev accuracy with comparable inference cost. If embeddings capture no useful structure, accuracy will not improve, indicating the approach fails.
```

```python
"""label-embed: per-position byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid (tuned on validation split)
EMBED_DIMS = [8, 16, 32]
LEARNING_RATES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
VAL_EPOCHS = 10          # epochs for hyperparameter search
FULL_EPOCHS = 20         # epochs for final training
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_loop(X, y, d, D, C, lr, epochs, rng, W_emb, W_lin, b_lin):
    """SGD with Adam on embeddings and linear layer. Returns updated params."""
    n = X.shape[0]
    # Adam state
    m_emb, v_emb = np.zeros_like(W_emb), np.zeros_like(W_emb)
    m_lin, v_lin = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_b, v_b = np.zeros_like(b_lin), np.zeros_like(b_lin)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)
            # Forward: embeddings -> concat -> linear
            emb_seq = W_emb[xb]  # (B, d, D)
            flat = emb_seq.reshape(B, d*D)  # (B, d*D)
            logits = flat @ W_lin + b_lin   # (B, C)
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B
            # Gradients
            grad_lin = flat.T @ probs + L2 * W_lin          # (d*D, C)
            grad_b = probs.sum(axis=0)                      # (C,)
            grad_flat = probs @ W_lin.T                     # (B, d*D)
            grad_emb_seq = grad_flat.reshape(B, d, D)       # (B, d, D)
            # Accumulate embedding gradients per (position, byte)
            grad_emb = np.zeros_like(W_emb)                 # (d, 256, D)
            np.add.at(grad_emb, (np.arange(d)[:,None], xb.T), grad_emb_seq.transpose(1,0,2))
            grad_emb += L2 * W_emb
            # Adam updates
            t += 1
            for param, grad, m, v in [
                (W_emb, grad_emb, m_emb, v_emb),
                (W_lin, grad_lin, m_lin, v_lin),
                (b_lin, grad_b, m_b, v_b)
            ]:
                m *= beta1; m += (1-beta1) * grad
                v *= beta2; v += (1-beta2) * grad * grad
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return W_emb, W_lin, b_lin

def evaluate(X, y, d, D, C, W_emb, W_lin, b_lin):
    emb_seq = W_emb[X]                 # (N, d, D)
    flat = emb_seq.reshape(-1, d*D)
    logits = flat @ W_lin + b_lin
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    val_start = int(n * 0.8)
    train_idx = np.arange(val_start)
    val_idx = np.arange(val_start, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_acc = -1.0
    best_params = None
    rng_base = np.random.default_rng(SEED)

    for D in EMBED_DIMS:
        for lr in LEARNING_RATES:
            rng = np.random.default_rng(SEED)  # reset for each config
            # Init embeddings: (d, 256, D)
            W_emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
            # Init linear: (d*D, C)
            W_lin = rng.normal(0, math.sqrt(2.0/(d*D)), size=(d*D, C)).astype(np.float32)
            b_lin = np.zeros(C, dtype=np.float32)
            # Train on validation split
            W_emb, W_lin, b_lin = train_loop(X_train, y_train, d, D, C, lr, VAL_EPOCHS, rng,
                                             W_emb, W_lin, b_lin)
            acc = evaluate(X_val, y_val, d, D, C, W_emb, W_lin, b_lin)
            if acc > best_acc:
                best_acc = acc
                best_params = (D, lr, W_emb.copy(), W_lin.copy(), b_lin.copy())

    D, lr, W_emb, W_lin, b_lin = best_params
    # Retrain on full data
    rng = np.random.default_rng(SEED)
    W_emb, W_lin, b_lin = train_loop(X, y, d, D, C, lr, FULL_EPOCHS, rng,
                                     W_emb, W_lin, b_lin)

    # Save weights: magic, d, D, C, reserved; then embeddings, W_lin, b_lin (all float32 LE)
    magic = 0x454D424C  # "LMBE" in LE
    header = struct.pack("<5i", magic, d, D, C, 0)
    emb_bytes = W_emb.astype("<f4").tobytes()
    lin_bytes = W_lin.astype("<f4").tobytes()
    b_bytes = b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + lin_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *out_len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *out_len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void write_rec(FILE *f, const unsigned char *data, uint32_t len) {
    unsigned char hdr[4] = {len & 255, len>>8 & 255, len>>16 & 255, len>>24};
    fwrite(hdr, 1, 4, f);
    fwrite(data, 1, len, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long wlen, ilen = 0;
    long N = atol(argv[3]);
    int32_t *hdr = (int32_t *)slurp(argv[1], &wlen);
    if (wlen < 20 || hdr[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], D = hdr[2], C = hdr[3];
    if (d != 16 || C != 65 || D <= 0 || D > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    size_t expected = 20 +
        (size_t)d * 256 * D * sizeof(float) +
        (size_t)d * D * C * sizeof(float) +
        (size_t)C * sizeof(float);
    if (wlen != (long)expected) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    const float *emb = (const float *)(hdr + 5);
    const float *W = emb + (size_t)d * 256 * D;
    const float *b = W + (size_t)d * D * C;

    unsigned char *input = N ? slurp(argv[2], &ilen) : NULL;
    long pos = 0;
    float *emb_vec = (float *)malloc((size_t)d * D * sizeof(float));
    float *scores = (float *)malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *rec = next_rec(input, ilen, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "record %ld: expected %d bytes, got %u\n", i, d, k); return 3; }
        // Build concatenated embedding vector
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((size_t)p * 256 + rec[p]) * D;
            memcpy(emb_vec + p * D, e, D * sizeof(float));
        }
        // Linear layer: scores = b + emb_vec @ W
        memcpy(scores, b, C * sizeof(float));
        for (int row = 0; row < d * D; row++) {
            float v = emb_vec[row];
            if (v == 0.0f) continue;
            const float *wrow = W + (size_t)row * C;
            for (int c = 0; c < C; c++) scores[c] += v * wrow[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        write_rec(stdout, &best, 1);
    }
    return 0;
}
```