```hypothesis
Family: label-rnn — a small recurrent network with byte embeddings, a single tanh RNN layer, and a linear output.
Why: Character prediction benefits from sequential modeling; an RNN captures n-gram statistics and longer-range dependencies that a linear model misses. The 16-byte context is short enough for a tiny RNN to train quickly on 24k samples.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost (Ir/item) exceeds label-linear by >2×, the hypothesis is not validated.
```

```python
"""label-rnn: tiny recurrent network for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x524E4E31, d, E, H, C; float32 embed[256][E], W_xh[E][H], W_hh[H][H], b_h[H], W_hy[H][C], b_y[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"E": 32, "H": 64, "lr": 0.01, "l2": 1e-5, "batch": 64, "epochs": 20},
    {"E": 16, "H": 32, "lr": 0.01, "l2": 1e-5, "batch": 64, "epochs": 20},
    {"E": 64, "H": 128, "lr": 0.005, "l2": 1e-5, "batch": 64, "epochs": 15},
]
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_data(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    return X[:-tune_n], y[:-tune_n], X[-tune_n:], y[-tune_n:]

def init_params(E, H, C, rng):
    # embeddings: uniform [-0.1, 0.1]
    emb = rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32)
    # Xavier for RNN weights
    W_xh = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / (E + H))).astype(np.float32)
    W_hh = (rng.uniform(-1, 1, (H, H)) * math.sqrt(6.0 / (2 * H))).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_hy = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b_y = np.zeros(C, np.float32)
    return emb, W_xh, W_hh, b_h, W_hy, b_y

def forward(X_batch, emb, W_xh, W_hh, b_h, W_hy, b_y):
    """X_batch: (B, d) uint8. Returns logits (B, C), hidden states (B, d, H) for BPTT."""
    B, d = X_batch.shape
    H = b_h.shape[0]
    # Embedding lookup
    emb_seq = emb[X_batch]  # (B, d, E)
    h = np.zeros((B, H), np.float32)
    h_seq = np.zeros((B, d, H), np.float32)
    for t in range(d):
        h = np.tanh(emb_seq[:, t] @ W_xh + h @ W_hh + b_h)
        h_seq[:, t] = h
    logits = h @ W_hy + b_y
    return logits, h_seq, emb_seq

def loss_and_grad(X_batch, y_batch, emb, W_xh, W_hh, b_h, W_hy, b_y, l2):
    B, d = X_batch.shape
    H = b_h.shape[0]
    C = b_y.shape[0]
    logits, h_seq, emb_seq = forward(X_batch, emb, W_xh, W_hh, b_h, W_hy, b_y)
    # Softmax + cross-entropy
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()
    # L2 regularization
    loss += 0.5 * l2 * (np.sum(emb**2) + np.sum(W_xh**2) + np.sum(W_hh**2) + np.sum(W_hy**2))
    # Gradients
    dlogits = probs
    dlogits[np.arange(B), y_batch] -= 1.0
    dlogits /= B
    # Output layer
    dW_hy = h_seq[:, -1].T @ dlogits + l2 * W_hy
    db_y = dlogits.sum(axis=0)
    dh = dlogits @ W_hy.T  # (B, H)
    # BPTT
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    demb = np.zeros_like(emb)
    for t in reversed(range(d)):
        dtanh = dh * (1.0 - h_seq[:, t]**2)
        dW_xh += emb_seq[:, t].T @ dtanh
        if t == 0:
            dW_hh += np.zeros((B, H)).T @ dtanh  # h_{-1} = 0
        else:
            dW_hh += h_seq[:, t-1].T @ dtanh
        db_h += dtanh.sum(axis=0)
        dh = dtanh @ W_hh.T
        # Embedding gradients
        np.add.at(demb, X_batch[:, t], dtanh @ W_xh.T)
    dW_xh += l2 * W_xh
    dW_hh += l2 * W_hh
    return loss, demb, dW_xh, dW_hh, db_h, dW_hy, db_y

def train_model(X_tr, y_tr, X_val, y_val, hp, rng):
    E, H, C = hp["E"], hp["H"], 65
    emb, W_xh, W_hh, b_h, W_hy, b_y = init_params(E, H, C, rng)
    # Adam state
    params = [emb, W_xh, W_hh, b_h, W_hy, b_y]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X_tr.shape[0]
    batch = hp["batch"]
    lr = hp["lr"]
    l2 = hp["l2"]
    for epoch in range(hp["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, g_emb, g_W_xh, g_W_hh, g_b_h, g_W_hy, g_b_y = loss_and_grad(
                Xb, yb, emb, W_xh, W_hh, b_h, W_hy, b_y, l2)
            grads = [g_emb, g_W_xh, g_W_hh, g_b_h, g_W_hy, g_b_y]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    val_logits, _, _ = forward(X_val, emb, W_xh, W_hh, b_h, W_hy, b_y)
    val_pred = val_logits.argmax(axis=1)
    acc = (val_pred == y_val).mean()
    return acc, params

def pack_weights(d, E, H, C, params):
    emb, W_xh, W_hh, b_h, W_hy, b_y = params
    magic = 0x524E4E31  # 'RNN1'
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (emb.astype("<f4").tobytes() +
            W_xh.astype("<f4").tobytes() +
            W_hh.astype("<f4").tobytes() +
            b_h.astype("<f4").tobytes() +
            W_hy.astype("<f4").tobytes() +
            b_y.astype("<f4").tobytes())
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # Split tuning set (last 20% in file order)
    X_tr, y_tr, X_val, y_val = split_data(X, y, 0.2)
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_hp = None
    for hp in GRID:
        acc, params = train_model(X_tr, y_tr, X_val, y_val, hp, rng)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_hp = hp
    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    _, best_params = train_model(X, y, X[:1], y[:1], best_hp, rng)  # dummy val
    out = pack_weights(d, best_hp["E"], best_hp["H"], C, best_params)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-rnn inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x524E4E31, d, E, H, C (int32);
 * float32 embed[256][E], W_xh[E][H], W_hh[H][H], b_h[H], W_hy[H][C], b_y[C]. */
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
    if (nw < 20 || hd[0] != 0x524E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)E * H + (long)H * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *embed = (const float *)(hd + 5);
    const float *W_xh = embed + 256L * E;
    const float *W_hh = W_xh + (long)E * H;
    const float *b_h = W_hh + (long)H * H;
    const float *W_hy = b_h + H;
    const float *b_y = W_hy + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *h_new = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int t = 0; t < d; t++) {
            const float *e = embed + (long)xi[t] * E;
            // h_new = tanh(e @ W_xh + h @ W_hh + b_h)
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int k2 = 0; k2 < E; k2++) sum += e[k2] * W_xh[(long)k2 * H + j];
                for (int k2 = 0; k2 < H; k2++) sum += h[k2] * W_hh[(long)k2 * H + j];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }
        // logits = h @ W_hy + b_y
        for (int c = 0; c < C; c++) {
            float sum = b_y[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_hy[(long)j * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```