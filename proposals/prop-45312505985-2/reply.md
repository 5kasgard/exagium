```hypothesis
A two-layer MLP with 128 hidden units and position-specific embeddings will capture non-linear interactions between the 16 context bytes, outperforming the linear model (0.3375) on dev accuracy. If dev accuracy does not exceed 0.34, the hypothesis is false.
```

```python
"""MLP family for next-byte prediction: 16 categorical bytes -> 128 ReLU -> 65 classes.
Uses position-specific embeddings (only observed values), Adam, early stopping on last 20% tuning split,
then retrains on full data. Exports weights in label-linear format (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

H = 128
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X_bytes.shape[0]
    d = X_bytes.shape[1]

    # tuning split: last 20%
    tune_size = int(0.2 * n)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:-tune_size], idx[-tune_size:]

    # build per-position encodings on training data only
    enc = []
    for j in range(d):
        vals = np.unique(X_bytes[train_idx, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)

    def encode(X):
        return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    Z_train = encode(X_bytes[train_idx])
    Z_tune = encode(X_bytes[tune_idx])
    y_train = y[train_idx]
    y_tune = y[tune_idx]

    k = Z_train.shape[1]
    rng = np.random.default_rng(SEED)

    # init weights
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Z):
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return a, h, logits

    def loss_grad(logits, y_batch):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_batch)), y_batch] -= 1.0
        return probs.astype(np.float32) / len(y_batch)

    best_acc = -1.0
    best_params = None
    best_epoch = 0
    patience_cnt = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
    t = 0

    for epoch in range(max_epochs):
        order = rng.permutation(len(train_idx))
        for s in range(0, len(train_idx), BATCH):
            batch_idx = order[s:s+BATCH]
            z, yy = Z_train[batch_idx], y_train[batch_idx]

            a, h, logits = forward(z)
            g = loss_grad(logits, yy)

            # backprop
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = (g @ W2.T) * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)

            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        _, _, logits_tune = forward(Z_tune)
        pred = logits_tune.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    # retrain on full data for best_epoch epochs
    Z_full = encode(X_bytes)
    params = [p.copy() for p in best_params]  # start from best
    W1, b1, W2, b2 = params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s+BATCH]
            z, yy = Z_full[batch_idx], y[batch_idx]
            a, h, logits = forward(z)
            g = loss_grad(logits, yy)
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = (g @ W2.T) * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    W1, b1, W2, b2 = params

    # build per-position lookup tables T_j = E_j @ W1_j
    off = 0
    T_list = []
    for E in enc:
        Vj = E.shape[1]
        Tj = E @ W1[off:off+Vj]
        T_list.append(Tj.astype(np.float32))
        off += Vj

    # pack weights: magic, d, H, C
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T_list).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```