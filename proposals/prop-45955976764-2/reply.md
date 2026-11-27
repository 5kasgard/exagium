```hypothesis
Family: label-embed-mlp. The task is next-character prediction from 16 bytes of context. A linear model on one-hot bytes (label-linear) cannot capture interactions between positions and characters. Using learned per-position embeddings reduces parameters and groups similar characters, while a small ReLU hidden layer adds non-linear context mixing. This should beat label-linear's 33.75% accuracy with comparable or lower inference cost (embedding lookups + two small matmuls). Failure: dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear's 7359.9.
```
```python
"""label-embed-mlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (E=embedding dim, H=hidden size) chosen on last 20% of train.in,
then retrained on all train. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(50, ceil(600000/n))."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50
EMBED_DIMS = [4, 8, 16]
HIDDEN_SIZES = [32, 64, 128]
MAGIC = 0x4C424D45  # 'L','B','M','E' little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, E, H, rng):
    # embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # W1: [d*E, H], b1: [H]
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C], b2: [C]
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(emb, W1, b1, W2, b2, X, d, E, H):
    n = X.shape[0]
    # embeddings lookup and concat
    flat = np.zeros((n, d * E), np.float32)
    for j in range(d):
        flat[:, j*E:(j+1)*E] = emb[j, X[:, j]]
    # hidden
    h = flat @ W1 + b1
    h = np.maximum(h, 0, dtype=np.float32)
    # output
    out = h @ W2 + b2
    return out, h, flat

def loss_grad(out, y, C):
    n = out.shape[0]
    out = out - out.max(axis=1, keepdims=True)
    exp = np.exp(out, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    grad = (probs / n).astype(np.float32)
    return grad

def evaluate(emb, W1, b1, W2, b2, X, y, d, E, H):
    out, _, _ = forward(emb, W1, b1, W2, b2, X, d, E, H)
    pred = out.argmax(axis=1)
    return (pred == y).mean()

def train_model(X, y, d, C, E, H):
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    emb, W1, b1, W2, b2 = init_params(d, C, E, H, rng)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            out, h, flat = forward(emb, W1, b1, W2, b2, xb, d, E, H)
            g_out = loss_grad(out, yb, C)
            # grad W2, b2
            g_W2 = h.T @ g_out + L2 * W2
            g_b2 = g_out.sum(0)
            # grad h
            g_h = g_out @ W2.T
            g_h[h == 0] = 0.0
            # grad W1, b1
            g_W1 = flat.T @ g_h + L2 * W1
            g_b1 = g_h.sum(0)
            # grad flat -> grad emb
            g_flat = g_h @ W1.T
            g_emb = np.zeros_like(emb)
            for j in range(d):
                g_emb[j] += g_flat[:, j*E:(j+1)*E].T @ np.eye(256, dtype=np.float32)[xb[:, j]]
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # update emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # update W1, b1
            m_W1 = beta1 * m_W1 + (1 - beta1) * g_W1
            v_W1 = beta2 * v_W1 + (1 - beta2) * g_W1 * g_W1
            m_b1 = beta1 * m_b1 + (1 - beta1) * g_b1
            v_b1 = beta2 * v_b1 + (1 - beta2) * g_b1 * g_b1
            W1 -= LR * (m_W1 / (1 - beta1 ** t)) / (np.sqrt(v_W1 / (1 - beta2 ** t)) + eps)
            b1 -= LR * (m_b1 / (1 - beta1 ** t)) / (np.sqrt(v_b1 / (1 - beta2 ** t)) + eps)
            # update W2, b2
            m_W2 = beta1 * m_W2 + (1 - beta1) * g_W2
            v_W2 = beta2 * v_W2 + (1 - beta2) * g_W2 * g_W2
            m_b2 = beta1 * m_b2 + (1 - beta1) * g_b2
            v_b2 = beta2 * v_b2 + (1 - beta2) * g_b2 * g_b2
            W2 -= LR * (m_W2 / (1 - beta1 ** t)) / (np.sqrt(v_W2 / (1 - beta2 ** t)) + eps)
            b2 -= LR * (m_b2 / (1 - beta1 ** t)) / (np.sqrt(v_b2 / (1 - beta2 ** t)) + eps)
    return emb, W1, b1, W2, b2

def save_weights(path, params, d, C, E, H):
    emb, W1, b1, W2, b2 = params
    # flatten embeddings to [d, 256, E] row-major
    emb_flat = emb.astype("<f4").tobytes()
    W1_flat = W1.astype("<f4").tobytes()
    b1_flat = b1.astype("<f4").tobytes()
    W2_flat = W2.astype("<f4").tobytes()
    b2_flat = b2.astype("<f4").tobytes()
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    with open(path, "wb") as f:
        f.write(header)
        f.write(emb_flat)
        f.write(W1_flat)
        f.write(b1_flat)
        f.write(W2_flat)
        f.write(b2_flat)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n_all = X_all.shape[0]
    split = int(0.8 * n_all)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]
    best_acc = -1.0
    best_E = EMBED_DIMS[0]
    best_H = HIDDEN_SIZES[0]
    best_params = None
    for E in EMBED_DIMS:
        for H in HIDDEN_SIZES:
            params = train_model(X_tr, y_tr, d, C, E, H)
            acc = evaluate(*params, X_tune, y_tune, d, E, H)
            if acc > best_acc:
                best_acc = acc
                best_E = E
                best_H = H
                best_params = params
    # retrain on all data
    final_params = train_model(X_all, y_all, d, C, best_E, best_H)
    save_weights(sys.argv[4], final_params, d, C, best_E, best_H)

if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (0..64).
 * Weights format: magic(0x4C424D45), d, E, H, C (int32), then float32 arrays:
 *   emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all row-major, little-endian). */
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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *out = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // hidden layer: h = ReLU( sum_j emb[j][xi[j]] @ W1 + b1 )
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < E; v++) sum += e[v] * W1[((long)j * E + v) * H + u];
                h[u] += sum;
            }
        }
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        // output: out = h @ W2 + b2
        for (int c = 0; c < C; c++) out[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            for (int c = 0; c < C; c++) out[c] += hu * W2[(long)u * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (out[c] > out[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```