```hypothesis
We propose a two-layer MLP with per-position embeddings (family "label-mlpemb"). Each of the 16 context bytes is mapped to a 32-dimensional embedding, concatenated, passed through a 128-unit ReLU hidden layer, then to 65 output classes. This captures interactions between positions that the linear model misses, while keeping parameter count ~200k and inference fast. If dev accuracy does not exceed 0.34, the model does not justify its extra complexity.
```

```python
"""Two-layer MLP with per-position embeddings for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16 positions × 256 vocab → 32-dim embeddings → concat (512) → ReLU(128) → Linear(65).
Adam lr=1e-3, L2=1e-4, batch=64, epochs=min(30, ceil(600000/n))."""
import json, math, struct, sys

import numpy as np

E = 32          # embedding size per position
H = 128         # hidden layer size
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 30


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
    C = card["classes"]
    d = card["input"]["bytes"]          # 16
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        # Vectorized embedding lookup: (d, batch, E) -> (batch, d, E)
        emb_out = emb[np.arange(d)[:, None], X_batch.T].transpose(1, 0, 2)
        concat = emb_out.reshape(X_batch.shape[0], d * E)
        hidden = concat @ W1 + b1
        hidden_relu = np.maximum(hidden, 0, dtype=np.float32)
        logits = hidden_relu @ W2 + b2
        return logits, hidden_relu, concat

    def loss_grad(logits, y_batch):
        batch = logits.shape[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        return (probs / batch).astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            logits, hidden_relu, concat = forward(X_batch)
            g_logits = loss_grad(logits, y_batch)

            # Output layer
            g_W2 = hidden_relu.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)

            # Hidden layer
            g_hidden = g_logits @ W2.T
            g_hidden[hidden_relu == 0] = 0
            g_W1 = concat.T @ g_hidden + L2 * W1
            g_b1 = g_hidden.sum(axis=0)

            # Embeddings
            g_concat = g_hidden @ W1.T
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_emb[j], X_batch[:, j], g_concat[:, j * E:(j + 1) * E])
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

    # Save weights: magic, d, E, H, C
    magic = 0x4D4C5032  # "MLP2"
    out = struct.pack("<5i", magic, d, E, H, C)
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
/* Two-layer MLP with per-position embeddings: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (0..64). */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding lookup + concatenation */
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, sizeof(float) * E);
        }

        /* Hidden layer: ReLU(concat * W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k_idx = 0; k_idx < d * E; k_idx++)
                sum += concat[k_idx] * W1[(long)k_idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer: hidden * W2 + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++)
                sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```