```hypothesis
Family: label-embedmlp — position-specific byte embeddings (16 positions × 256 vocab × E) fed into a 1-hidden-layer ReLU MLP (H units) with softmax output.
Why: Character-level language modeling benefits from distributed representations; embeddings capture semantic similarity between characters (e.g., 'a'/'b', 'A'/'a') and position-specific n-gram statistics more parameter-efficiently than one-hot linear models. The 16-byte context is short enough that a shallow net suffices.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 15k Ir/item (2× label-linear).
```
```python
"""label-embedmlp: position-specific embeddings + 1-hidden-layer ReLU MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: int32 magic=0x454D4231, d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension per position
H = 64          # hidden units
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

def softmax_crossentropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits. logits: (B, C), y: (B,)"""
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, d, E)
            emb_seq = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, E)
            flat = emb_seq.reshape(B, d * E)                       # (B, d*E)
            hidden_pre = flat @ W1 + b1                            # (B, H)
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)   # ReLU
            logits = hidden @ W2 + b2                              # (B, C)

            # Loss gradient
            g_logits = softmax_crossentropy_grad(logits, yb)       # (B, C)

            # Backprop
            g_W2 = hidden.T @ g_logits + L2 * W2                   # (H, C)
            g_b2 = g_logits.sum(axis=0)                            # (C,)
            g_hidden = g_logits @ W2.T                             # (B, H)
            g_hidden[hidden_pre <= 0] = 0.0                        # ReLU grad
            g_W1 = flat.T @ g_hidden + L2 * W1                     # (d*E, H)
            g_b1 = g_hidden.sum(axis=0)                            # (H,)
            g_flat = g_hidden @ W1.T                               # (B, d*E)
            g_emb_seq = g_flat.reshape(B, d, E)                    # (B, d, E)

            # Embedding gradients: scatter-add
            g_emb = np.zeros_like(emb)
            for j in range(d):
                bytes_j = xb[:, j]                                 # (B,)
                np.add.at(g_emb[j], bytes_j, g_emb_seq[:, j, :])   # (256, E)
            g_emb += L2 * emb

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fold embeddings into first-layer lookup tables T[d][256][H]
    # T[j, byte, h] = sum_e emb[j, byte, e] * W1[j*E + e, h]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j*E:(j+1)*E, :]  # (256, E) @ (E, H) -> (256, H)

    # Pack weights
    # header: magic, d, E, H, C (5 int32 = 20 bytes)
    header = struct.pack("<5i", 0x454D4231, d, E, H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index 0..C-1).
 * Weight format: int32 magic=0x454D4231, d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc  = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            for (int c = 0; c < C; c++) sc[c] += ah * W2[(long)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```