```hypothesis
Family: label-embmlp (embedding MLP with sum-pooling). The linear model uses 4096 one-hot features (16×256) which overfits on 24k examples. This model learns 16 position-specific embedding tables (256×D), sums the 16 vectors (16·D adds), then passes through an optional ReLU hidden layer (D→H) and a linear output (H→65). The low-rank factorization reduces parameters (≈16·256·D + D·H + H·65) and captures character similarities; the hidden layer adds nonlinear interactions. With D=64, H=64 inference cost is ~1k adds + 8k mults, comparable to linear's 1k adds. If dev accuracy does not exceed 0.3375 (label-linear) the family fails.
```
```python
"""label-embmlp: embedding MLP with sum-pooling for fixed-size byte sequences.
train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single-threaded, stdlib+numpy only.
Grid-searches (D, H, lr, L2) on last 20% of train.in, then retrains on all data.
Exports weights: magic 0x454D424C, d, D, H, C; then 16×256×D embeddings (float32);
if H>0: W1[D×H], b1[H], W2[H×C], b2[C]; else: W_out[D×C], b_out[C]."""
import json, math, struct, sys
import numpy as np

SEED = 12345
GRID = {
    'D': [32, 64, 96],
    'H': [0, 64, 96],
    'lr': [1e-3, 3e-3],
    'L2': [1e-4, 1e-3],
}
EPOCHS_TUNE = 15
EPOCHS_FULL = 40
BATCH = 128
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, D, H, C, rng):
    emb = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    if H:
        W1 = rng.normal(0, math.sqrt(2.0 / D), (D, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return emb, W1, b1, W2, b2
    else:
        W = rng.normal(0, math.sqrt(2.0 / D), (D, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        return emb, W, b, None, None

def forward(X, emb, W1, b1, W2, b2, H):
    """X: (N, d) uint8. Returns logits (N, C)."""
    N, d = X.shape
    # sum embeddings
    vec = np.zeros((N, emb[0].shape[1]), np.float32)
    for j in range(d):
        vec += emb[j][X[:, j]]
    if H:
        h = np.maximum(vec @ W1 + b1, 0, dtype=np.float32)
        return h @ W2 + b2
    else:
        return vec @ W1 + b1  # W1 is W_out, b1 is b_out

def loss_and_grad(X, y, emb, W1, b1, W2, b2, H, L2, C):
    N, d = X.shape
    # forward
    vec = np.zeros((N, emb[0].shape[1]), np.float32)
    for j in range(d):
        vec += emb[j][X[:, j]]
    if H:
        a = vec @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = vec @ W1 + b1
    # softmax + cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(N), y]).mean()
    # L2 regularization
    reg = 0.0
    for e in emb:
        reg += (e * e).sum()
    if H:
        reg += (W1 * W1).sum() + (W2 * W2).sum()
    else:
        reg += (W1 * W1).sum()
    loss += 0.5 * L2 * reg
    # backward
    grad_logits = probs
    grad_logits[np.arange(N), y] -= 1.0
    grad_logits /= N
    if H:
        grad_W2 = h.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(0)
        grad_h = grad_logits @ W2.T
        grad_h[a <= 0] = 0.0
        grad_W1 = vec.T @ grad_h + L2 * W1
        grad_b1 = grad_h.sum(0)
        grad_vec = grad_h @ W1.T
    else:
        grad_W1 = vec.T @ grad_logits + L2 * W1
        grad_b1 = grad_logits.sum(0)
        grad_vec = grad_logits @ W1.T
    grad_emb = [np.zeros_like(e) for e in emb]
    for j in range(d):
        np.add.at(grad_emb[j], X[:, j], grad_vec)
        grad_emb[j] += L2 * emb[j]
    if H:
        return loss, grad_emb, grad_W1, grad_b1, grad_W2, grad_b2
    else:
        return loss, grad_emb, grad_W1, grad_b1, None, None

def adam_update(params, grads, states, t, lr):
    for i, (p, g) in enumerate(zip(params, grads)):
        if g is None:
            continue
        m, v = states[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def accuracy(X, y, emb, W1, b1, W2, b2, H):
    logits = forward(X, emb, W1, b1, W2, b2, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_split(X_tr, y_tr, X_val, y_val, d, C, D, H, lr, L2, epochs, batch, rng):
    emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    params = [W1, b1] if not H else [W1, b1, W2, b2]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X_tr.shape[0]
    steps_per_epoch = max(1, n // batch)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, g_emb, g_W1, g_b1, g_W2, g_b2 = loss_and_grad(Xb, yb, emb, W1, b1, W2, b2, H, L2, C)
            t += 1
            # update embeddings
            for j in range(d):
                m, v = states_emb[j]
                m *= 0.9; m += 0.1 * g_emb[j]
                v *= 0.999; v += 0.001 * g_emb[j] * g_emb[j]
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                emb[j] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            # update other params
            if H:
                adam_update([W1, b1, W2, b2], [g_W1, g_b1, g_W2, g_b2], states, t, lr)
            else:
                adam_update([W1, b1], [g_W1, g_b1], states, t, lr)
    val_acc = accuracy(X_val, y_val, emb, W1, b1, W2, b2, H)
    return emb, W1, b1, W2, b2, val_acc

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for D in GRID['D']:
        for H in GRID['H']:
            for lr in GRID['lr']:
                for L2 in GRID['L2']:
                    cfg_rng = np.random.default_rng(SEED + hash((D, H, lr, L2)) % 2**32)
                    emb, W1, b1, W2, b2, acc = train_split(X_tr, y_tr, X_val, y_val, d, C, D, H, lr, L2, EPOCHS_TUNE, BATCH, cfg_rng)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (D, H, lr, L2)
                        best_params = (emb, W1, b1, W2, b2)
                    print(f"D={D} H={H} lr={lr} L2={L2} val_acc={acc:.4f}", flush=True)

    D, H, lr, L2 = best_cfg
    print(f"Best config: D={D} H={H} lr={lr} L2={L2} val_acc={best_acc:.4f}", flush=True)
    # retrain on full data
    full_rng = np.random.default_rng(SEED + 1)
    emb, W1, b1, W2, b2, _ = train_split(X, y, X_val[:1], y_val[:1], d, C, D, H, lr, L2, EPOCHS_FULL, BATCH, full_rng)

    # pack weights
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<4i", magic, d, D, H, C)
    out = bytearray(header)
    for e in emb:
        out += e.astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D424C, d, D, H, C (int32 each).
 * Then d * 256 * D float32 embeddings (position-major).
 * If H>0: W1[D*H], b1[H], W2[H*C], b2[C] (float32).
 * Else: W_out[D*C], b_out[C] (float32). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * D * 4;
    if (H) {
        expected += (long)D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    } else {
        expected += (long)D * C * 4 + C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *W_out, *b_out;
    if (H) {
        W1 = emb + (long)d * 256 * D;
        b1 = W1 + (long)D * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W_out = emb + (long)d * 256 * D;
        b_out = W_out + (long)D * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // zero vec
        for (int u = 0; u < D; u++) vec[u] = 0.0f;
        // sum embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) vec[u] += e[u];
        }
        if (H) {
            // hidden = relu(vec @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int u = 0; u < D; u++) sum += vec[u] * W1[(long)u * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = vec @ W_out + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int u = 0; u < D; u++) sum += vec[u] * W_out[(long)u * C + c];
                logits[c] = sum;
            }
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```