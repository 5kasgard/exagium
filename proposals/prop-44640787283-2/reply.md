```hypothesis
Family: label-embed (embedding + one ReLU layer). Learns dense 8-d embeddings per byte position, then a 64-unit hidden layer. Should capture character interactions better than linear one-hot (0.3375) with far fewer parameters than GBDT, and inference stays ~10k Ir/item. Fails if dev accuracy ≤ 0.34 or Ir/item > 15000.
```
```python
"""label-embed: embedding + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16 positions × 256 vocab → 8-d embeddings → concat (128) → ReLU(64) → linear(65).
Adam lr=1e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x454D4244, d, E, H, C; then float32 arrays:
  emb[d*256*E], W1[d*E*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension
H = 64         # hidden units
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4244  # 'EMBD'

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

    # Combined embedding matrix: (d*256, E)
    emb = rng.normal(0, 1/math.sqrt(E), size=(d*256, E)).astype(np.float32)

    # Hidden layer weights
    fan_in = d * E
    W1 = rng.normal(0, math.sqrt(2.0 / fan_in), size=(fan_in, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)

    # Output layer
    W2 = rng.normal(0, 1/math.sqrt(H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute position offsets for embedding indices
    pos_offsets = np.arange(d) * 256  # shape (d,)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # ---- Forward ----
            # Gather embeddings: for each position, indices = xb[:,j] + pos_offsets[j]
            # We'll build concat by looping over positions (d=16 small)
            concat = np.zeros((B, d * E), dtype=np.float32)
            for j in range(d):
                indices = xb[:, j] + pos_offsets[j]   # (B,)
                concat[:, j*E:(j+1)*E] = emb[indices]

            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B   # (B, C)

            # ---- Backward ----
            # Output layer
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T
            dhidden[hidden_pre <= 0] = 0.0

            # Hidden layer
            dW1 = concat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dconcat = dhidden @ W1.T

            # Embeddings: scatter-add gradients per position
            demb = np.zeros_like(emb)
            for j in range(d):
                indices = xb[:, j] + pos_offsets[j]
                np.add.at(demb, indices, dconcat[:, j*E:(j+1)*E])
            demb += L2 * emb

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ---- Export weights ----
    # Order: emb, W1, b1, W2, b2
    total_floats = d*256*E + d*E*H + H + H*C + C
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D4244, d, E, H, C; float32 arrays:
 *   emb[d*256*E], W1[d*E*H], b1[H], W2[H*C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concat embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int c = 0; c < d * E; c++) sum += concat[c] * W1[(long)c * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            for (int c = 0; c < C; c++) scores[c] += hv * W2[(long)h * C + c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```