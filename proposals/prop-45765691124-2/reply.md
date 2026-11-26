```hypothesis
Family: label-embed-mlp (one shared embedding per position, one ReLU hidden layer).
Why: The task is character-level language modeling with 16-byte context. One-hot linear models ignore semantic similarity between bytes; embeddings capture that and reduce parameters. A hidden layer adds non-linearity for interactions. This should beat label-linear (0.3375) while staying faster than label-gbdt.
Failure indicator: dev accuracy ≤ 0.34 or inference cost > 2× label-linear.
```

```python
"""Embedding + one hidden layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position byte embeddings (D=32) -> concat -> ReLU(H=128) -> linear(65).
Adam lr=1e-3, L2=1e-4, batch=256, epochs=30."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 32          # embedding dimension
H = 128         # hidden units
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 256
EPOCHS = 30
MAGIC = 0x454D424C  # "LMBE" in little-endian

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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: [d, 256, D]
    emb = (rng.normal(0, 1/math.sqrt(D), size=(d, 256, D))).astype(np.float32)
    # W1: [d*D, H], b1: [H]
    W1 = (rng.normal(0, math.sqrt(2.0/(d*D)), size=(d*D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C], b2: [C]
    W2 = (rng.normal(0, math.sqrt(2.0/H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: [B, d] uint8
        B = X_batch.shape[0]
        # Embedding lookup
        x_emb = emb[X_batch, np.arange(256)]  # This is wrong: need proper indexing
        # Correct indexing:
        x_emb = np.zeros((B, d, D), dtype=np.float32)
        for j in range(d):
            x_emb[:, j, :] = emb[j, X_batch[:, j], :]
        x_concat = x_emb.reshape(B, d*D)
        h_pre = x_concat @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        return x_concat, h_pre, h, logits

    def loss_grad(logits, y_batch):
        B = logits.shape[0]
        # softmax cross-entropy gradient
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B
        return probs  # dL/dlogits

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            x_emb = np.zeros((B, d, D), dtype=np.float32)
            for j in range(d):
                x_emb[:, j, :] = emb[j, Xb[:, j], :]
            x_concat = x_emb.reshape(B, d*D)
            h_pre = x_concat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Gradient on logits
            g_logits = loss_grad(logits, yb)  # [B, C]

            # Gradients
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_h = g_logits @ W2.T
            g_h_pre = g_h * (h_pre > 0)
            g_W1 = x_concat.T @ g_h_pre + L2 * W1
            g_b1 = g_h_pre.sum(axis=0)
            # Gradient on x_concat
            g_x_concat = g_h_pre @ W1.T  # [B, d*D]
            g_x_emb = g_x_concat.reshape(B, d, D)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb[j], Xb[:, j], g_x_emb[:, j, :])
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Magic, d, D, H, C
    header = struct.pack("<5i", MAGIC, d, D, H, C)
    # Embeddings [d, 256, D] -> flatten in order d, 256, D
    emb_bytes = emb.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding + one hidden layer MLP inference.
 * ./infer weights.bin in.rec N
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20
        + (long)d * 256 * D * 4
        + (long)d * D * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_concat = malloc(sizeof(float) * d * D);
    float *h_pre = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(x_concat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: h_pre = x_concat @ W1 + b1
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) {
                sum += x_concat[v] * W1[(long)v * H + u];
            }
            h_pre[u] = sum;
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h @ W2 + b2
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