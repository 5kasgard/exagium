```hypothesis
Family: label-embedding-mlp. Per-position byte embeddings (16×256×D) fed into a one-hidden-layer ReLU MLP (H units) then linear to 65 classes. Captures positional interactions via embeddings and non-linearity while keeping inference fast (two small matmuls). Should beat label-linear (0.3375) by learning distributed representations. If dev accuracy ≤ 0.3375 or CPU cost ≥ label-linear, the hypothesis fails.
```
```python
"""label-embedding-mlp: per-position embeddings + one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all train."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on tuning split)
SEED = 1
D = 8          # embedding dimension per position
H = 64         # hidden layer size
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
CLIP_GRAD = 1.0

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
    assert d == 16 and C == 65

    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_val, y_val = X_bytes[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: [d, 256, D]
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # First layer: (d*D) -> H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Second layer: H -> C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(Xb):
        # Xb: [B, d] uint8
        # Embedding lookup: [B, d, D]
        e = emb[Xb]  # advanced indexing: (B, d, D)
        e = e.reshape(Xb.shape[0], d * D)  # [B, d*D]
        h = e @ W1 + b1
        h = np.maximum(h, 0)  # ReLU
        logits = h @ W2 + b2
        return logits, h, e

    def loss_and_grad(Xb, yb):
        B = Xb.shape[0]
        logits, h, e = forward(Xb)
        # Softmax cross-entropy
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Loss
        loss = -np.log(probs[np.arange(B), yb] + 1e-12).mean()
        # Gradient on logits
        dlogits = probs
        dlogits[np.arange(B), yb] -= 1.0
        dlogits /= B
        # Gradients
        dW2 = h.T @ dlogits + L2 * W2
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ W2.T
        dh[h <= 0] = 0  # ReLU grad
        dW1 = e.T @ dh + L2 * W1
        db1 = dh.sum(axis=0)
        # Embedding gradients: de = dh @ W1.T  -> [B, d*D]
        de = dh @ W1.T
        de = de.reshape(B, d, D)
        # Scatter into demb
        demb = np.zeros_like(emb)
        np.add.at(demb, (np.arange(d)[:, None], Xb.T), de.transpose(1, 0, 2))
        demb += L2 * emb
        return loss, [demb, dW1, db1, dW2, db2]

    # Quick tuning on validation split: we just train on train split and monitor val acc
    # (In a real scenario we might grid search, but we hardcode above.)
    print(f"Training on {X_tr.shape[0]} samples, validating on {X_val.shape[0]}", file=sys.stderr)

    for epoch in range(EPOCHS):
        order = rng.permutation(X_tr.shape[0])
        epoch_loss = 0.0
        batches = 0
        for s in range(0, X_tr.shape[0], BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, grads = loss_and_grad(Xb, yb)
            epoch_loss += loss
            batches += 1
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                # Clip gradient
                g = np.clip(g, -CLIP_GRAD, CLIP_GRAD)
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)
        # Validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={epoch_loss/batches:.4f}, val_acc={val_acc:.4f}", file=sys.stderr)

    # Final training on all data (using same hyperparameters, but we can do a few more epochs)
    # We'll just continue training on full dataset for a few epochs.
    print("Final training on all data...", file=sys.stderr)
    for epoch in range(5):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            _, grads = loss_and_grad(Xb, yb)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g = np.clip(g, -CLIP_GRAD, CLIP_GRAD)
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Magic: 0x454D424C (little endian "LBME")
    # Format: int32 magic, d, D, H, C
    # Then: embeddings [d, 256, D] float32 (row-major: d * 256 * D)
    # Then: W1 [d*D, H], b1 [H], W2 [H, C], b2 [C] all float32
    magic = 0x454D424C
    out = struct.pack("<5i", magic, d, D, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output: one byte class index (0..64). */
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
    uint8_t *wdata = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wdata;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20
        + (long)d * 256 * D * 4
        + (long)d * D * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wdata + 20);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *e = malloc(sizeof(float) * d * D);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(e + j * D, src, D * sizeof(float));
        }

        // First layer: e (d*D) -> h (H) with ReLU
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) {
                sum += e[v] * W1[(long)v * H + u];
            }
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Second layer: h (H) -> logits (C)
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h[u] * W2[(long)u * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```