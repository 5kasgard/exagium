```hypothesis
Family: label-embed (positional embeddings summed then linear). 
Why: Captures per-position byte statistics with low-rank factorization, reducing parameters vs one-hot linear while modeling position-specific effects. Should beat label-linear on accuracy with similar or lower inference cost (D=32 gives ~133k params vs 266k). 
Failure: If dev accuracy <= label-linear (0.3375) or instructions/item not lower.
```

```python
"""label-embed: positional embeddings summed then linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrain on all.
Exported weights: magic 0x454D4244, int32 d,D,C; float32 embeddings[d][256][D], W[C][D], b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
GRID = [
    {"D": 16, "lr": 1e-3, "l2": 1e-4, "epochs_tune": 10, "epochs_full": 50, "batch": 64},
    {"D": 32, "lr": 3e-3, "l2": 1e-4, "epochs_tune": 10, "epochs_full": 50, "batch": 64},
    {"D": 64, "lr": 1e-3, "l2": 1e-5, "epochs_tune": 10, "epochs_full": 50, "batch": 64},
    {"D": 32, "lr": 1e-2, "l2": 1e-4, "epochs_tune": 10, "epochs_full": 50, "batch": 64},
]
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, D, C, rng):
    # embeddings: list of d arrays (256, D)
    embeddings = [rng.normal(0, 0.01, size=(256, D)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 0.01, size=(C, D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    return embeddings, W, b

def forward(embeddings, W, b, X):
    # X: (n, d) uint8
    n, d = X.shape
    D = embeddings[0].shape[1]
    rep = np.zeros((n, D), dtype=np.float32)
    for j in range(d):
        rep += embeddings[j][X[:, j]]
    logits = rep @ W.T + b  # (n, C)
    return logits, rep

def loss_and_grad(logits, y, rep, embeddings, W, b, X, l2):
    n, C = logits.shape
    d, D = X.shape[1], rep.shape[1]
    # softmax cross-entropy
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # dL/dlogits
    # gradients
    dW = probs.T @ rep + l2 * W
    db = probs.sum(axis=0)
    drep = probs @ W  # (n, D)
    dembeddings = [np.zeros_like(e) for e in embeddings]
    for j in range(d):
        np.add.at(dembeddings[j], X[:, j], drep)
        dembeddings[j] += l2 * embeddings[j]
    loss = -np.log(np.maximum(probs[np.arange(n), y] * n + 1e-12, 1e-12)).mean()  # approximate
    return loss, dembeddings, dW, db

def train_model(X_train, y_train, X_val, y_val, config, rng):
    d, C = X_train.shape[1], len(np.unique(y_train))
    D = config["D"]
    lr, l2, batch, epochs = config["lr"], config["l2"], config["batch"], config["epochs"]
    embeddings, W, b = init_params(d, D, C, rng)
    # Adam state
    m_emb = [np.zeros_like(e) for e in embeddings]
    v_emb = [np.zeros_like(e) for e in embeddings]
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X_train.shape[0]
    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb, yb = X_train[idx], y_train[idx]
            logits, rep = forward(embeddings, W, b, Xb)
            loss, demb, dW, db = loss_and_grad(logits, yb, rep, embeddings, W, b, Xb, l2)
            t += 1
            # Adam update for embeddings
            for j in range(d):
                m_emb[j] = beta1 * m_emb[j] + (1 - beta1) * demb[j]
                v_emb[j] = beta2 * v_emb[j] + (1 - beta2) * demb[j] * demb[j]
                m_hat = m_emb[j] / (1 - beta1 ** t)
                v_hat = v_emb[j] / (1 - beta2 ** t)
                embeddings[j] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # Adam update for W, b
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW * dW
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db * db
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        # validation
        if X_val is not None:
            val_logits, _ = forward(embeddings, W, b, X_val)
            val_pred = val_logits.argmax(axis=1)
            val_acc = (val_pred == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = ([e.copy() for e in embeddings], W.copy(), b.copy())
    if X_val is not None and best_params is not None:
        embeddings, W, b = best_params
    return embeddings, W, b, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_cfg = None
    best_acc = -1.0
    for cfg in GRID:
        _, _, _, val_acc = train_model(X_train, y_train, X_tune, y_tune, cfg, rng)
        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg
    print(f"Best config: {best_cfg}, val_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data
    embeddings, W, b, _ = train_model(X, y, None, None, best_cfg, rng)
    # save weights
    D = best_cfg["D"]
    magic = 0x454D4244
    header = struct.pack("<4i", magic, d, D, C)
    # embeddings: d * 256 * D floats
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in embeddings)
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D4244, int32 d,D,C; float32 embeddings[d][256][D], W[C][D], b[C]. */
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
    if (d <= 0 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)C * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * D;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * D);
    float *logits = (float *)malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < D; u++) acc[u] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) acc[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *w = W + (long)c * D;
            for (int u = 0; u < D; u++) sum += acc[u] * w[u];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```