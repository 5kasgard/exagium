```hypothesis
Family: label-embmlp. Uses a shared byte embedding (size E) and position-specific projections to a hidden layer (size H) with ReLU, then a linear output. This captures pairwise interactions between context positions through the hidden layer while sharing statistical strength across byte values via embeddings. It should beat label-linear (0.3375) on dev accuracy because the non-linear hidden layer models positional interactions that a linear model cannot. If dev accuracy does not exceed label-linear, or if inference cost per item exceeds label-linear by more than 2×, the hypothesis fails.
```
```python
"""label-embmlp: shared embedding + position-specific projection to hidden ReLU layer + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, d, E, H, C; float32 embedding[256][E], pos_proj[16][E][H], h_bias[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
E = 32          # embedding size
H = 64          # hidden size
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 10   # early stopping patience on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n_all)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_proj = (rng.uniform(-1, 1, (d, E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    h_bias = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, pos_proj, h_bias, W2, b2]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, params):
        emb, pos_proj, h_bias, W2, b2 = params
        n = X.shape[0]
        h = np.zeros((n, H), np.float32)
        for j in range(d):
            e = emb[X[:, j]]                 # (n, E)
            h += e @ pos_proj[j]             # (n, H)
        h += h_bias
        h = np.maximum(h, 0)                 # ReLU
        logits = h @ W2 + b2                 # (n, C)
        return logits, h

    def loss_grad(logits, y):
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        grad_logits = (probs / n).astype(np.float32)
        return grad_logits

    def backward(X, h, grad_logits, params):
        emb, pos_proj, h_bias, W2, b2 = params
        n = X.shape[0]
        # grad for W2, b2
        grad_W2 = h.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(axis=0)
        # grad for h (after ReLU)
        grad_h = grad_logits @ W2.T
        grad_h[h == 0] = 0.0
        # grad for h_bias
        grad_h_bias = grad_h.sum(axis=0)
        # grad for pos_proj and emb
        grad_pos_proj = np.zeros_like(pos_proj)
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            e = emb[X[:, j]]                 # (n, E)
            grad_pos_proj[j] = e.T @ grad_h + L2 * pos_proj[j]
            grad_emb_j = grad_h @ pos_proj[j].T
            np.add.at(grad_emb, X[:, j], grad_emb_j)
        return [grad_emb, grad_pos_proj, grad_h_bias, grad_W2, grad_b2]

    def accuracy(X, y, params):
        logits, _ = forward(X, params)
        pred = logits.argmax(axis=1)
        return (pred == y).mean()

    best_val_acc = -1.0
    best_params = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, h = forward(Xb, params)
            grad_logits = loss_grad(logits, yb)
            grads = backward(Xb, h, grad_logits, params)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        val_acc = accuracy(X_val, y_val, params)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # retrain on full data for best_epoch epochs
    params = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_all[idx], y_all[idx]
            logits, h = forward(Xb, params)
            grad_logits = loss_grad(logits, yb)
            grads = backward(Xb, h, grad_logits, params)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # save weights
    emb, pos_proj, h_bias, W2, b2 = params
    magic = 0x454D424C  # "LMBE"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += pos_proj.astype("<f4").tobytes()
    out += h_bias.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output is one byte: predicted class index.
 * Weights format: magic 0x454D424C, int32 d, E, H, C; float32 embedding[256][E], pos_proj[d][E][H], h_bias[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *pos_proj = emb + 256L * E;
    const float *h_bias = pos_proj + (long)d * E * H;
    const float *W2 = h_bias + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = h_bias
        memcpy(h, h_bias, H * sizeof(float));
        // accumulate position projections
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            const float *P = pos_proj + ((long)j * E * H);
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int e_idx = 0; e_idx < E; e_idx++) {
                    sum += e[e_idx] * P[(long)e_idx * H + u];
                }
                h[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        // logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```