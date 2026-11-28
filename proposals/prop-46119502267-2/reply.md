```hypothesis
We propose an embedding-MLP family: each of the 16 context bytes gets a position-specific embedding of size H (default 64). The embeddings are summed, passed through a ReLU hidden layer of size H, then a linear output layer to 65 classes. This captures non-linear interactions between positions while keeping inference at ~16*H + H*65 operations. It should beat label-linear (0.3375) because character-level language modeling benefits from non-linear context mixing. If dev accuracy ≤ 0.34, the model fails to improve.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5
H_CANDIDATES = [32, 64, 128]  # hidden sizes to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    probs[np.arange(n), y] -= 1.0
    grad = probs / n
    return loss, grad.astype(np.float32)

def forward(X, params, H):
    # X: (n, d) uint8
    d = X.shape[1]
    n = X.shape[0]
    E, b1, W2, b2 = params
    # E: (d, 256, H)
    h = np.zeros((n, H), dtype=np.float32)
    for j in range(d):
        h += E[j, X[:, j]]  # (n, H)
    h += b1
    np.maximum(h, 0, out=h)  # ReLU
    logits = h @ W2 + b2
    return logits, h

def train_model(X_tr, y_tr, X_val, y_val, H, rng):
    d = X_tr.shape[1]
    C = 65
    # init embeddings
    E = (rng.uniform(-1, 1, (d, 256, H)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1
    best_params = None
    best_epoch = 0
    n_tr = X_tr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            logits, h = forward(xb, params, H)
            loss, grad_logits = softmax_cross_entropy(logits, yb)
            # backprop
            # grad_logits: (B, C)
            # dW2 = h.T @ grad_logits
            # db2 = grad_logits.sum(0)
            # dh = grad_logits @ W2.T * (h > 0)
            # dE[j][v] += dh[i] for each sample i where xb[i,j]==v
            # db1 = dh.sum(0)
            B = xb.shape[0]
            dW2 = h.T @ grad_logits + L2 * W2
            db2 = grad_logits.sum(0) + L2 * b2
            dh = (grad_logits @ W2.T) * (h > 0)
            db1 = dh.sum(0) + L2 * b1
            dE = np.zeros_like(E)
            for j in range(d):
                # accumulate dh for each byte value
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        dE[j, v] += dh[mask].sum(0)
                dE[j] += L2 * E[j]
            grads = [dE, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # validation
        logits_val, _ = forward(X_val, params, H)
        pred = logits_val.argmax(axis=1)
        val_acc = (pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    return best_params, best_epoch, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert d == 16 and C == 65
    n = X.shape[0]
    val_size = n // 5
    tr_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    rng = np.random.default_rng(SEED)
    best_H = H_CANDIDATES[0]
    best_val_acc = -1
    best_params = None
    best_epoch = 0
    for H in H_CANDIDATES:
        # reset seed for each candidate to ensure deterministic comparison
        candidate_rng = np.random.default_rng(SEED + H)
        params, epoch, val_acc = train_model(X_tr, y_tr, X_val, y_val, H, candidate_rng)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_H = H
            best_params = params
            best_epoch = epoch
    # retrain on full data with best_H for best_epoch epochs
    rng_full = np.random.default_rng(SEED + best_H + 1000)
    E, b1, W2, b2 = best_params
    # We'll just continue training from best_params for the same number of epochs? Simpler: train from scratch for best_epoch epochs.
    # But we already have best_params trained on subset. To be safe, we train from scratch on full data for best_epoch epochs.
    # Re-initialize with same seed and train for best_epoch epochs (no early stopping).
    E = (rng_full.uniform(-1, 1, (d, 256, best_H)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    b1 = np.zeros(best_H, np.float32)
    W2 = (rng_full.uniform(-1, 1, (best_H, C)) * math.sqrt(6.0 / best_H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(1, best_epoch + 1):
        order = rng_full.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h = forward(xb, params, best_H)
            loss, grad_logits = softmax_cross_entropy(logits, yb)
            B = xb.shape[0]
            dW2 = h.T @ grad_logits + L2 * W2
            db2 = grad_logits.sum(0) + L2 * b2
            dh = (grad_logits @ W2.T) * (h > 0)
            db1 = dh.sum(0) + L2 * b1
            dE = np.zeros_like(E)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        dE[j, v] += dh[mask].sum(0)
                dE[j] += L2 * E[j]
            grads = [dE, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    E, b1, W2, b2 = params
    # write weights
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, best_H, C)
    out += E.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *b1 = E + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(h, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) h[u] += e[u];
        }
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        memcpy(scores, b2, C * sizeof(float));
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += hu * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```