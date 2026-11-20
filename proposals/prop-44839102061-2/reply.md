```hypothesis
Family: label-emb16 (byte embeddings + one hidden ReLU layer, precomputed per-position contributions).
Why: The linear model treats each position independently; embeddings let the model learn similarities between bytes (e.g., letters vs punctuation) and the hidden layer captures cross-position interactions. Precomputing (position,byte)→hidden contributions keeps inference at ~16·H adds + H·C MACs (H=64), similar latency to linear but with more capacity.
Failure: dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 2× label-linear.
```
```python
"""label-emb16: byte embeddings + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D4231, d, E, H, C; float32 hidden_contrib[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension
H = 64         # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: shared 256 x E, positional d x E
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_emb = (rng.uniform(-1, 1, (d, E)) * math.sqrt(6.0 / d)).astype(np.float32)

    # Hidden layer: (d*E) x H
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H x C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup + positional
        # (B, d, E)
        x_emb = emb[X_batch] + pos_emb[None, :, :]  # broadcasting
        # Flatten to (B, d*E)
        flat = x_emb.reshape(B, d * E)
        # Hidden pre-activation
        h_pre = flat @ W1 + b1
        h = np.maximum(h_pre, 0)
        # Logits
        logits = h @ W2 + b2
        return logits, h_pre, h, flat

    def softmax_grad(logits, yy):
        # logits (B, C), yy (B,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h_pre, h, flat = forward(Xb)
            g_logits = softmax_grad(logits, yb)  # (B, C)

            # Gradients
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_h = g_logits @ W2.T
            g_h_pre = g_h * (h_pre > 0)
            g_W1 = flat.T @ g_h_pre + L2 * W1
            g_b1 = g_h_pre.sum(axis=0)
            # Back to embeddings
            g_flat = g_h_pre @ W1.T  # (B, d*E)
            g_emb_flat = g_flat.reshape(BATCH, d, E)
            g_emb = np.zeros_like(emb)
            g_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                bytes_j = Xb[:, j]
                np.add.at(g_emb, bytes_j, g_emb_flat[:, j, :])
                g_pos_emb[j] += g_emb_flat[:, j, :].sum(axis=0)

            grads = [g_emb, g_pos_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute hidden contributions per (position, byte)
    # hidden_contrib[d][256][H]
    hidden_contrib = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        for v in range(256):
            vec = emb[v] + pos_emb[j]  # (E,)
            # Multiply by W1 block for position j
            W1_block = W1[j * E:(j + 1) * E, :]  # (E, H)
            hidden_contrib[j, v] = vec @ W1_block

    # Pack weights
    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
    out += hidden_contrib.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb16 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D4231, d, E, H, C (int32 each),
 *   float32 hidden_contrib[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *hidden_contrib = (const float *)(hd + 5);
    const float *b1 = hidden_contrib + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_pre = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden pre-activation = b1 + sum_j hidden_contrib[j][xi[j]]
        memcpy(h_pre, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *contrib = hidden_contrib + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) h_pre[h] += contrib[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (h_pre[h] < 0.0f) h_pre[h] = 0.0f;

        // Output layer
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = h_pre[h];
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2h[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h_pre); free(scores);
    return 0;
}
```