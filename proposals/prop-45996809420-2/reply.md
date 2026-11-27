```hypothesis
Family: label-embmlp (shared byte embeddings + one ReLU hidden layer).
Why: The 16-byte context has strong local character correlations; shared embeddings capture per-byte statistics across positions, and a small hidden layer models interactions. This should exceed linear accuracy (0.3375) with moderate cost (embeddings 256×8, hidden 32).
Failure: If dev accuracy ≤ 0.34 or Ir/item > 20000, the model is not a worthwhile improvement.
```

```python
"""label-embmlp: shared byte embeddings (E=8) + one ReLU hidden layer (H=32) for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses validation split (last 20%) for early stopping, then retrains on all data."""
import json, math, struct, sys

import numpy as np

SEED = 1
E = 8          # embedding dimension
H = 32         # hidden units
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # split: last 20% for validation
    val_start = int(n * 0.8)
    X_tr, y_tr = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: 256 x E
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # first layer: (d*E) x H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # second layer: H x C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (batch, d) uint8
        # embed: (batch, d, E)
        emb_x = emb[Xb]                     # (batch, d, E)
        flat = emb_x.reshape(len(Xb), d * E) # (batch, d*E)
        h = flat @ W1 + b1
        h = np.maximum(h, 0, dtype=np.float32)
        logits = h @ W2 + b2
        return logits, h, flat

    def loss_grad(logits, yb):
        # softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return probs / len(yb)   # dL/dlogits

    best_val_acc = -1.0
    best_params = None
    best_epoch = 0
    t = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            logits, h, flat = forward(xb)
            g_logits = loss_grad(logits, yb)          # (batch, C)
            # grad W2, b2
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            # grad hidden
            g_h = g_logits @ W2.T
            g_h[h == 0] = 0.0
            # grad W1, b1
            g_W1 = flat.T @ g_h + L2 * W1
            g_b1 = g_h.sum(axis=0)
            # grad embeddings
            g_flat = g_h @ W1.T                       # (batch, d*E)
            g_emb = np.zeros_like(emb)
            # scatter-add: for each position and batch, add g_flat slice to emb[byte]
            g_flat_3d = g_flat.reshape(len(xb), d, E)
            for j in range(d):
                bytes_j = xb[:, j]
                np.add.at(g_emb, bytes_j, g_flat_3d[:, j, :])
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch
        elif epoch - best_epoch >= PATIENCE:
            break

    # retrain on full data for best_epoch epochs
    emb, W1, b1, W2, b2 = best_params
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, flat = forward(xb)
            g_logits = loss_grad(logits, yb)
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_h = g_logits @ W2.T
            g_h[h == 0] = 0.0
            g_W1 = flat.T @ g_h + L2 * W1
            g_b1 = g_h.sum(axis=0)
            g_flat = g_h @ W1.T
            g_emb = np.zeros_like(emb)
            g_flat_3d = g_flat.reshape(len(xb), d, E)
            for j in range(d):
                bytes_j = xb[:, j]
                np.add.at(g_emb, bytes_j, g_flat_3d[:, j, :])
            g_emb += L2 * emb
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights: magic, d, E, H, C; then emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += params[0].astype("<f4").tobytes()   # emb
    out += params[1].astype("<f4").tobytes()   # W1
    out += params[2].astype("<f4").tobytes()   # b1
    out += params[3].astype("<f4").tobytes()   # W2
    out += params[4].astype("<f4").tobytes()   # b2
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // embed
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        // hidden = ReLU(flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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