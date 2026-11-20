```hypothesis
Family: label-embmlp (positional byte embeddings + one ReLU hidden layer).
Why: The linear model uses 16×256 one-hot features (4096 dim) directly to 65 classes, which cannot capture n-gram interactions. Embeddings (E=16) compress each byte to a dense vector, preserving positional information via separate tables per position. A hidden layer (H=64) then models interactions. This should exceed label-linear's 0.3375 accuracy while keeping inference cost modest (O(d·E·H + H·C) ≈ 16k multiply-adds). If dev accuracy ≤ 0.34 or Ir/item > 20,000, the hypothesis fails.
```

```python
"""label-embmlp: positional byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x454D424C), d, E, H, C (int32 each)
  embeddings[d][256][E] (float32)
  W1[d*E][H] (float32), b1[H] (float32)
  W2[H][C] (float32), b2[C] (float32)"""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 5    # early stopping on tuning split

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

    # split: last 20% for tuning (validation)
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    n_tr = X_tr.shape[0]

    # initialize embeddings: d x 256 x E
    emb = rng.normal(0, 1/math.sqrt(256), size=(d, 256, E)).astype(np.float32)
    # W1: (d*E) x H, b1: H
    W1 = rng.normal(0, math.sqrt(2/(d*E)), size=(d*E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C, b2: C
    W2 = rng.normal(0, math.sqrt(2/H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_params = None
    patience = 0

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # gather embeddings: (batch, d, E)
        emb_vecs = emb[Xb]  # advanced indexing: (batch, d, E)
        flat = emb_vecs.reshape(batch, d*E)  # (batch, d*E)
        hidden = flat @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU
        logits = hidden @ W2 + b2
        return logits, hidden, flat

    def loss_grad(logits, yb):
        batch = logits.shape[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(batch), yb] -= 1.0
        probs /= batch
        return probs

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx_b = order[s:s+BATCH]
            Xb, yb = X_tr[idx_b], y_tr[idx_b]
            logits, hidden, flat = forward(Xb)
            g_logits = loss_grad(logits, yb)  # (batch, C)

            # gradients
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(0)
            g_hidden = g_logits @ W2.T
            g_hidden[hidden == 0] = 0  # ReLU grad
            g_W1 = flat.T @ g_hidden + L2 * W1
            g_b1 = g_hidden.sum(0)
            # embedding grads: scatter-add
            g_emb = np.zeros_like(emb)
            g_flat = g_hidden @ W1.T  # (batch, d*E)
            g_emb_vecs = g_flat.reshape(BATCH, d, E)  # but batch may be smaller
            # use np.add.at for scatter
            for j in range(d):
                np.add.at(g_emb[j], Xb[:, j], g_emb_vecs[:, j, :])
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # retrain on full data with best params? We'll just use best_params and do a few more epochs on full data.
    # But the rules: choose settings on tuning split, then train on all of train.
    # We'll do a few epochs on full data starting from best_params.
    emb, W1, b1, W2, b2 = best_params
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(5):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_b = order[s:s+BATCH]
            Xb, yb = X[idx_b], y[idx_b]
            logits, hidden, flat = forward(Xb)
            g_logits = loss_grad(logits, yb)
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(0)
            g_hidden = g_logits @ W2.T
            g_hidden[hidden == 0] = 0
            g_W1 = flat.T @ g_hidden + L2 * W1
            g_b1 = g_hidden.sum(0)
            g_emb = np.zeros_like(emb)
            g_flat = g_hidden @ W1.T
            g_emb_vecs = g_flat.reshape(BATCH, d, E)
            for j in range(d):
                np.add.at(g_emb[j], Xb[:, j], g_emb_vecs[:, j, :])
            g_emb += L2 * emb
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    magic = 0x454D424C  # 'L','B','M','E' in little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += params[0].astype("<f4").tobytes()  # emb
    out += params[1].astype("<f4").tobytes()  # W1
    out += params[2].astype("<f4").tobytes()  # b1
    out += params[3].astype("<f4").tobytes()  # W2
    out += params[4].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format (little-endian):
 *   magic(0x454D424C), d, E, H, C (int32)
 *   embeddings[d][256][E] (float32)
 *   W1[d*E][H] (float32), b1[H] (float32)
 *   W2[H][C] (float32), b2[C] (float32) */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + (long)d * 256 * E
        + (long)d * E * H + H
        + (long)H * C + C;
    if (nw != 4 * expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, 4L * expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
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
        // gather embeddings into flat
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = flat + j * E;
            memcpy(dst, src, E * sizeof(float));
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
    free(flat); free(hidden); free(logits);
    return 0;
}
```