```hypothesis
Family: label-mlp-bn (bottleneck MLP with position-specific byte embeddings).
Why: The task is next-character prediction from 16 bytes. A linear model cannot capture interactions between positions. A two-layer MLP with a bottleneck (small per-position embeddings, larger hidden layer) can model pairwise and higher-order interactions efficiently. The bottleneck reduces parameters and compute compared to a full one-hot first layer (as in label-linear with H>0). Training with more epochs and tuned learning rate on the tuning split should yield higher accuracy than label-linear at similar or lower inference cost.
Failure: If dev accuracy <= 0.3375 or Ir/item > 10000, the hypothesis is disproven.
```

```python
"""Bottleneck MLP for next-byte prediction: position-specific embeddings -> hidden ReLU -> output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train."""
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning split search
GRID = [
    {"emb_dim": 16, "hidden_dim": 64, "lr": 1e-3, "l2": 1e-4, "epochs": 30},
    {"emb_dim": 32, "hidden_dim": 128, "lr": 3e-3, "l2": 1e-4, "epochs": 30},
    {"emb_dim": 64, "hidden_dim": 256, "lr": 1e-3, "l2": 1e-4, "epochs": 30},
]
SEED = 1
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

def init_params(d, emb_dim, hidden_dim, C, rng):
    # embeddings: list of d arrays (256, emb_dim)
    emb = [rng.normal(0, 0.01, (256, emb_dim)).astype(np.float32) for _ in range(d)]
    # W1: (d*emb_dim, hidden_dim)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * emb_dim)), (d * emb_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, np.float32)
    # W2: (hidden_dim, C)
    W2 = rng.normal(0, math.sqrt(2.0 / hidden_dim), (hidden_dim, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(emb, W1, b1, W2, b2, X):
    # X: (n, d) uint8
    n, d = X.shape
    # concatenate embeddings
    Z = np.zeros((n, d * emb[0].shape[1]), np.float32)
    for j in range(d):
        Z[:, j*emb[0].shape[1]:(j+1)*emb[0].shape[1]] = emb[j][X[:, j]]
    # hidden
    H = Z @ W1 + b1
    H = np.maximum(H, 0)
    # output
    logits = H @ W2 + b2
    return logits, Z, H

def softmax_cross_entropy(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad.astype(np.float32)

def train_step(emb, W1, b1, W2, b2, Z, H, grad_logits, lr, l2, opt_state, t):
    # opt_state: list of (m, v) for each param array in order: emb (flat), W1, b1, W2, b2
    # We'll flatten embeddings for optimization
    pass  # implemented below with explicit loops for clarity

def evaluate(emb, W1, b1, W2, b2, X, y):
    logits, _, _ = forward(emb, W1, b1, W2, b2, X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split tuning (last 20%)
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    n_train = train_X.shape[0]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None

    for cfg in GRID:
        emb_dim, hidden_dim = cfg["emb_dim"], cfg["hidden_dim"]
        lr, l2, epochs = cfg["lr"], cfg["l2"], cfg["epochs"]
        # re-seed for each config
        cfg_rng = np.random.default_rng(SEED + hash((emb_dim, hidden_dim)) % 2**32)
        emb, W1, b1, W2, b2 = init_params(d, emb_dim, hidden_dim, C, cfg_rng)

        # Adam state
        params = [emb, W1, b1, W2, b2]
        # flatten embeddings for optimizer
        emb_flat = np.concatenate([e.ravel() for e in emb])
        state = [
            (np.zeros_like(emb_flat), np.zeros_like(emb_flat)),
            (np.zeros_like(W1), np.zeros_like(W1)),
            (np.zeros_like(b1), np.zeros_like(b1)),
            (np.zeros_like(W2), np.zeros_like(W2)),
            (np.zeros_like(b2), np.zeros_like(b2)),
        ]
        t = 0
        for epoch in range(epochs):
            order = cfg_rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s:s+BATCH]
                xb, yb = train_X[idx], train_y[idx]
                # forward
                logits, Z, H = forward(emb, W1, b1, W2, b2, xb)
                loss, g_logits = softmax_cross_entropy(logits, yb)
                # backward
                # g_logits: (B, C)
                # dW2 = H.T @ g_logits
                dW2 = H.T @ g_logits + l2 * W2
                db2 = g_logits.sum(axis=0)
                # dH = g_logits @ W2.T * (H > 0)
                dH = (g_logits @ W2.T) * (H > 0)
                # dW1 = Z.T @ dH
                dW1 = Z.T @ dH + l2 * W1
                db1 = dH.sum(axis=0)
                # dZ = dH @ W1.T
                dZ = dH @ W1.T
                # dembeddings
                demb_flat = np.zeros_like(emb_flat)
                off = 0
                for j in range(d):
                    e = emb[j]
                    dj = dZ[:, j*emb_dim:(j+1)*emb_dim]
                    # scatter-add for each batch item
                    np.add.at(demb_flat, (xb[:, j][:, None] * emb_dim + np.arange(emb_dim)).ravel(), dj.ravel())
                # Adam updates
                grads = [demb_flat, dW1, db1, dW2, db2]
                t += 1
                for i, (p, gr) in enumerate(zip([emb_flat, W1, b1, W2, b2], grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                # update emb from flat
                off = 0
                for j in range(d):
                    emb[j] = emb_flat[off:off+256*emb_dim].reshape(256, emb_dim)
                    off += 256 * emb_dim

        acc = evaluate(emb, W1, b1, W2, b2, tune_X, tune_y)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # retrain on full data with best config
    cfg = best_cfg
    emb_dim, hidden_dim = cfg["emb_dim"], cfg["hidden_dim"]
    lr, l2, epochs = cfg["lr"], cfg["l2"], min(cfg["epochs"] * 2, MAX_EPOCHS)
    emb, W1, b1, W2, b2 = best_params
    emb_flat = np.concatenate([e.ravel() for e in emb])
    state = [
        (np.zeros_like(emb_flat), np.zeros_like(emb_flat)),
        (np.zeros_like(W1), np.zeros_like(W1)),
        (np.zeros_like(b1), np.zeros_like(b1)),
        (np.zeros_like(W2), np.zeros_like(W2)),
        (np.zeros_like(b2), np.zeros_like(b2)),
    ]
    rng = np.random.default_rng(SEED)
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits, Z, H = forward(emb, W1, b1, W2, b2, xb)
            loss, g_logits = softmax_cross_entropy(logits, yb)
            dW2 = H.T @ g_logits + l2 * W2
            db2 = g_logits.sum(axis=0)
            dH = (g_logits @ W2.T) * (H > 0)
            dW1 = Z.T @ dH + l2 * W1
            db1 = dH.sum(axis=0)
            dZ = dH @ W1.T
            demb_flat = np.zeros_like(emb_flat)
            off = 0
            for j in range(d):
                dj = dZ[:, j*emb_dim:(j+1)*emb_dim]
                np.add.at(demb_flat, (xb[:, j][:, None] * emb_dim + np.arange(emb_dim)).ravel(), dj.ravel())
            grads = [demb_flat, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip([emb_flat, W1, b1, W2, b2], grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            off = 0
            for j in range(d):
                emb[j] = emb_flat[off:off+256*emb_dim].reshape(256, emb_dim)
                off += 256 * emb_dim

    # write weights
    # magic: 0x4D4C5042 ("MLPB")
    out = struct.pack("<5i", 0x4D4C5042, d, emb_dim, hidden_dim, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Bottleneck MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x4D4C5042, d, emb_dim, hidden_dim, C;
 * embeddings[d][256][emb_dim] (float32), W1[d*emb_dim][hidden_dim], b1[hidden_dim],
 * W2[hidden_dim][C], b2[C]. All little-endian float32. */
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
    if (nw < 20 || hd[0] != 0x4D4C5042) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], hidden_dim = hd[3], C = hd[4];
    if (d != 16 || C != 65 || emb_dim <= 0 || hidden_dim <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * emb_dim
        + 4L * d * emb_dim * hidden_dim
        + 4L * hidden_dim
        + 4L * hidden_dim * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 5);
    const float *emb = ptr; ptr += d * 256 * emb_dim;
    const float *W1 = ptr; ptr += d * emb_dim * hidden_dim;
    const float *b1 = ptr; ptr += hidden_dim;
    const float *W2 = ptr; ptr += hidden_dim * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *Z = malloc(sizeof(float) * d * emb_dim);
    float *H = malloc(sizeof(float) * hidden_dim);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * emb_dim;
            float *z = Z + j * emb_dim;
            for (int k = 0; k < emb_dim; k++) z[k] = e[k];
        }

        // hidden layer: H = ReLU(Z @ W1 + b1)
        for (int h = 0; h < hidden_dim; h++) {
            float sum = b1[h];
            for (int z = 0; z < d * emb_dim; z++) sum += Z[z] * W1[z * hidden_dim + h];
            H[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: logits = H @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < hidden_dim; h++) sum += H[h] * W2[h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(Z); free(H); free(logits);
    return 0;
}
```