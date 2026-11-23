```hypothesis
Family: label-emb (embedding + shallow MLP). Each of the 14 byte features gets a learned 8-d embedding; embeddings are summed and passed through a 16-unit ReLU layer then a 2-class linear head. This captures feature interactions cheaply and should beat label-linear (0.8524) on dev while staying fast. Failure: dev accuracy ≤ 0.8524 or inference cost > 2× label-linear.
```
```python
"""label-emb: embedding + one ReLU layer for tabular byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2 = 3e-3, 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid searched on tuning split (last 20% of train.in)
EMB_GRID = [8, 16]
HIDDEN_GRID = [0, 16, 32]   # 0 = linear on embeddings
TUNE_EPOCHS = 10            # quick evaluation on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def adam_step(params, grads, state, t, lr=LR, l2=L2, b1=0.9, b2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        g = g + l2 * p
        m, v = state[i]
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)   # (B, C)

def forward_emb(X, emb, W1, b1, W2, b2, H):
    # X: (B, d) uint8
    B, d = X.shape
    E = emb.shape[2]
    # Sum embeddings: (B, E)
    h = np.zeros((B, E), np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    if H == 0:
        logits = h @ W1 + b1   # W1: (E, C), b1: (C,)
        return logits, h, None
    else:
        pre = h @ W1 + b1      # W1: (E, H), b1: (H,)
        hidden = np.maximum(pre, 0)
        logits = hidden @ W2 + b2   # W2: (H, C), b2: (C,)
        return logits, h, (pre, hidden)

def backward_emb(X, emb, W1, b1, W2, b2, H, h, hidden_cache, grad_logits):
    B, d = X.shape
    E = emb.shape[2]
    grad_emb = np.zeros_like(emb)
    if H == 0:
        grad_W1 = h.T @ grad_logits
        grad_b1 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ W1.T
    else:
        pre, hidden = hidden_cache
        grad_W2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ W2.T
        grad_pre = grad_hidden * (pre > 0)
        grad_W1 = h.T @ grad_pre
        grad_b1 = grad_pre.sum(axis=0)
        grad_h = grad_pre @ W1.T
    # Distribute grad_h to embeddings
    for j in range(d):
        np.add.at(grad_emb[j], X[:, j], grad_h)
    if H == 0:
        return grad_emb, grad_W1, grad_b1, None, None, None, None
    else:
        return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2, None, None

def evaluate(X, y, emb, W1, b1, W2, b2, H):
    logits, _, _ = forward_emb(X, emb, W1, b1, W2, b2, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_config(X_tr, y_tr, X_val, y_val, d, C, E, H, rng, epochs):
    # Initialize
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    if H == 0:
        W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, C))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        W2 = b2 = None
        params = [emb, W1, b1]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    else:
        W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X_tr.shape[0]
    t = 0
    best_val = -1.0
    best_params = None
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            logits, h, hidden_cache = forward_emb(xb, emb, W1, b1, W2, b2, H)
            grad_logits = softmax_cross_entropy_grad(logits, yb)
            grads = backward_emb(xb, emb, W1, b1, W2, b2, H, h, hidden_cache, grad_logits)
            t += 1
            adam_step(params, grads, state, t)
        # Validation
        val_acc = evaluate(X_val, y_val, emb, W1, b1, W2, b2, H)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    tune_size = n // 5
    tr_idx = np.arange(n - tune_size)
    val_idx = np.arange(n - tune_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_overall = -1.0
    best_cfg = None
    best_params = None

    for E in EMB_GRID:
        for H in HIDDEN_GRID:
            cfg_rng = np.random.default_rng(SEED + E * 100 + H)
            val_acc, params = train_config(X_tr, y_tr, X_val, y_val, d, C, E, H, cfg_rng, TUNE_EPOCHS)
            if val_acc > best_overall:
                best_overall = val_acc
                best_cfg = (E, H)
                best_params = params

    # Retrain on full data with best config
    E, H = best_cfg
    final_rng = np.random.default_rng(SEED + 9999)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    _, final_params = train_config(X, y, X_val, y_val, d, C, E, H, final_rng, epochs)

    # Unpack final params
    emb = final_params[0]
    if H == 0:
        W1, b1 = final_params[1], final_params[2]
        W2 = b2 = None
    else:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]

    # Write weights.bin
    # Magic: 0x454D4231 ("EMB1")
    # Header: magic, d, E, H, C (5 int32)
    # Then embeddings: d * 256 * E float32
    # Then if H==0: W1 (E*C), b1 (C)
    # Else: W1 (E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D4231
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += emb.astype("<f4").tobytes()
    if H == 0:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
    else:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights format: magic(0x454D4231), d, E, H, C (int32 each),
 *   emb[d][256][E] (float32),
 *   if H==0: W1[E][C], b1[C]
 *   else: W1[E][H], b1[H], W2[H][C], b2[C] */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t head_sz = 20 + emb_sz * 4;
    size_t rest_sz;
    if (H == 0) rest_sz = (size_t)E * C + C;
    else rest_sz = (size_t)E * H + H + (size_t)H * C + C;
    if (nw != (long)(head_sz + rest_sz * 4)) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + (H == 0 ? (size_t)E * C : (size_t)E * H);
    const float *W2 = b1 + (H == 0 ? C : H);
    const float *b2 = W2 + (H == 0 ? 0 : (size_t)H * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int e = 0; e < E; e++) sum_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += eptr[e];
        }

        if (H == 0) {
            // Linear: logits = sum_emb @ W1 + b1
            for (int c = 0; c < C; c++) {
                float acc = b1[c];
                for (int e = 0; e < E; e++) acc += sum_emb[e] * W1[(size_t)e * C + c];
                logits[c] = acc;
            }
        } else {
            // Hidden layer: hidden = relu(sum_emb @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float acc = b1[h];
                for (int e = 0; e < E; e++) acc += sum_emb[e] * W1[(size_t)e * H + h];
                hidden[h] = acc > 0.0f ? acc : 0.0f;
            }
            // Output: logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float acc = b2[c];
                for (int h = 0; h < H; h++) acc += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = acc;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```