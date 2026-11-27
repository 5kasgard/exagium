```hypothesis
Family: label-embmlp (per-position byte embeddings + one-hidden-layer MLP).
Why: The 16-byte context has strong positional patterns; embeddings reduce the 4096 one-hot dimensions to a dense 16*E representation, capturing similarity between characters. A small ReLU layer then models non-linear interactions. This should surpass label-linear's 0.3375 accuracy while using fewer multiply-adds (16*E*H + H*C vs 4096*C). 
Failure: dev accuracy ≤ 0.34 or instructions/item ≥ label-linear's 7360.
```

```python
"""label-embmlp: per-position embeddings + one-hidden-layer ReLU MLP.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data.
Weights format (little-endian):
  magic(4) d(4) E(4) H(4) C(4)
  embed[16][256][E] float32
  W1[d*E][H] float32, b1[H] float32
  W2[H][C] float32, b2[C] float32
"""
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
GRID = {
    'E': [4, 8, 16],
    'H': [32, 64, 128],
    'lr': [1e-3, 3e-3],
    'l2': [1e-4, 1e-5],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # Split for tuning
    X_tr, y_tr, X_val, y_val = train_val_split(X, y)
    n_tr, n_val = X_tr.shape[0], X_val.shape[0]

    best_acc = -1.0
    best_cfg = None
    best_params = None

    rng = np.random.default_rng(SEED)

    for E in GRID['E']:
        for H in GRID['H']:
            for lr in GRID['lr']:
                for l2 in GRID['l2']:
                    # Initialize parameters
                    # embeddings: 16 x 256 x E
                    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
                    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
                    b1 = np.zeros(H, np.float32)
                    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
                    b2 = np.zeros(C, np.float32)
                    params = [emb, W1, b1, W2, b2]
                    # Adam state
                    m = [np.zeros_like(p) for p in params]
                    v = [np.zeros_like(p) for p in params]
                    t = 0
                    beta1, beta2, eps = 0.9, 0.999, 1e-8

                    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
                    for _ in range(epochs):
                        order = rng.permutation(n_tr)
                        for s in range(0, n_tr, BATCH):
                            idx = order[s:s + BATCH]
                            xb = X_tr[idx]   # (B, d)
                            yb = y_tr[idx]   # (B,)
                            B = xb.shape[0]

                            # Forward
                            # embeddings: for each position, gather E-dim vector
                            emb_vecs = emb[np.arange(d)[:, None], xb.T]  # (d, B, E)
                            emb_cat = emb_vecs.transpose(1, 0, 2).reshape(B, d * E)  # (B, d*E)
                            h_pre = emb_cat @ W1 + b1
                            h = np.maximum(h_pre, 0)
                            logits = h @ W2 + b2

                            # Softmax + cross-entropy gradient
                            logits = logits - logits.max(axis=1, keepdims=True)
                            exp_logits = np.exp(logits)
                            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                            probs[np.arange(B), yb] -= 1.0
                            probs /= B

                            # Backprop
                            dW2 = h.T @ probs
                            db2 = probs.sum(axis=0)
                            dh = probs @ W2.T
                            dh[h_pre <= 0] = 0
                            dW1 = emb_cat.T @ dh
                            db1 = dh.sum(axis=0)
                            # Gradient wrt embeddings
                            demb_cat = dh @ W1.T  # (B, d*E)
                            demb = demb_cat.reshape(B, d, E).transpose(1, 0, 2)  # (d, B, E)
                            # Accumulate into demb_grad (d, 256, E)
                            demb_grad = np.zeros((d, 256, E), np.float32)
                            for j in range(d):
                                np.add.at(demb_grad[j], xb[:, j], demb[j])

                            # L2 regularization
                            dW1 += l2 * W1
                            dW2 += l2 * W2
                            # Note: no L2 on embeddings and biases for simplicity

                            grads = [demb_grad, dW1, db1, dW2, db2]

                            t += 1
                            for i, (p, gr) in enumerate(zip(params, grads)):
                                m[i] = beta1 * m[i] + (1 - beta1) * gr
                                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                                m_hat = m[i] / (1 - beta1 ** t)
                                v_hat = v[i] / (1 - beta2 ** t)
                                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    # Evaluate on validation
                    emb_vecs = emb[np.arange(d)[:, None], X_val.T]
                    emb_cat = emb_vecs.transpose(1, 0, 2).reshape(n_val, d * E)
                    h = np.maximum(emb_cat @ W1 + b1, 0)
                    logits = h @ W2 + b2
                    preds = logits.argmax(axis=1)
                    acc = accuracy(y_val, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (E, H, lr, l2)
                        best_params = [emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy()]

    # Retrain on all data with best config
    E, H, lr, l2 = best_cfg
    emb, W1, b1, W2, b2 = best_params
    # Re-initialize with best config but using the same seed? We'll continue training from the best params? 
    # Simpler: retrain from scratch on all data with best config (deterministic).
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))
    for _ in range(epochs):
        order = rng.permutation(X.shape[0])
        for s in range(0, X.shape[0], BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            emb_vecs = emb[np.arange(d)[:, None], xb.T]
            emb_cat = emb_vecs.transpose(1, 0, 2).reshape(B, d * E)
            h_pre = emb_cat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            dW2 = h.T @ probs
            db2 = probs.sum(axis=0)
            dh = probs @ W2.T
            dh[h_pre <= 0] = 0
            dW1 = emb_cat.T @ dh
            db1 = dh.sum(axis=0)
            demb_cat = dh @ W1.T
            demb = demb_cat.reshape(B, d, E).transpose(1, 0, 2)
            demb_grad = np.zeros((d, 256, E), np.float32)
            for j in range(d):
                np.add.at(demb_grad[j], xb[:, j], demb[j])

            dW1 += l2 * W1
            dW2 += l2 * W2

            grads = [demb_grad, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x454D424C  # "LBME"
    emb, W1, b1, W2, b2 = params
    # Validate shapes
    assert emb.shape == (d, 256, E)
    assert W1.shape == (d * E, H)
    assert b1.shape == (H,)
    assert W2.shape == (H, C)
    assert b2.shape == (C,)
    # Pack
    header = struct.pack("<5i", magic, d, E, H, C)
    body = emb.astype("<f4").tobytes()
    body += W1.astype("<f4").tobytes()
    body += b1.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes()
    body += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format (little-endian):
 *   magic(4) d(4) E(4) H(4) C(4)
 *   embed[d][256][E] float32
 *   W1[d*E][H] float32, b1[H] float32
 *   W2[H][C] float32, b2[C] float32
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_cat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_cat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(emb_cat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h; // column-major? W1 is (d*E) x H stored row-major? In train.py W1 shape (d*E, H) row-major.
            // Actually in train.py: W1 = (d*E, H) row-major. So W1[input_idx * H + h] is the weight.
            for (int idx = 0; idx < d * E; idx++) {
                sum += emb_cat[idx] * W1[idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```