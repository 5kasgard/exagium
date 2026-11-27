```hypothesis
Family: label-embed-mlp. Learns a per-position byte embedding (dim 8) followed by one ReLU hidden layer (32 units). This captures local interactions in the 16-byte context more effectively than linear one-hot models while keeping parameters modest (~156k). Expects to beat label-linear's 0.3375 accuracy with similar or lower inference cost. Failure if dev accuracy ≤ 0.34 or Ir/item > 7500.
```
```python
"""label-embed-mlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension per position
H = 32         # hidden layer size
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    V = 256                             # byte values

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: list of d arrays, each (V, E)
    emb = [(rng.standard_normal((V, E), dtype=np.float32) / math.sqrt(E)) for _ in range(d)]

    # Hidden layer: input size d*E, output H
    W1 = (rng.standard_normal((d * E, H), dtype=np.float32) * math.sqrt(2.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer: H -> C
    W2 = (rng.standard_normal((H, C), dtype=np.float32) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [W1, b1, W2, b2]
    emb_params = emb  # list of arrays
    all_params = emb_params + params

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in all_params]
    t = 0

    def softmax_grad(logits, targets):
        # logits: (batch, C), targets: (batch,)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return (probs / len(targets)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)
            batch_size = len(yb)

            # Embedding lookup and concatenation
            # For each position j, gather rows emb[j][xb[:,j]] -> (batch, E)
            emb_concat = np.concatenate([emb[j][xb[:, j]] for j in range(d)], axis=1)  # (batch, d*E)

            # Forward
            hidden_pre = emb_concat @ W1 + b1          # (batch, H)
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2                  # (batch, C)

            # Loss gradient
            g_logits = softmax_grad(logits, yb)        # (batch, C)

            # Backprop output layer
            g_W2 = hidden.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_hidden = g_logits @ W2.T                  # (batch, H)

            # ReLU gradient
            g_hidden_pre = g_hidden * (hidden_pre > 0)

            # Backprop hidden layer
            g_W1 = emb_concat.T @ g_hidden_pre + L2 * W1
            g_b1 = g_hidden_pre.sum(axis=0)
            g_emb_concat = g_hidden_pre @ W1.T          # (batch, d*E)

            # Split gradient for each embedding table
            g_emb = []
            for j in range(d):
                g_emb.append(g_emb_concat[:, j*E:(j+1)*E])

            # Accumulate gradients for all parameters
            grads = []
            for j in range(d):
                # Gradient for embedding table j: scatter-add
                g_table = np.zeros_like(emb[j])
                np.add.at(g_table, xb[:, j], g_emb[j])
                g_table += L2 * emb[j]
                grads.append(g_table)
            grads.extend([g_W1, g_b1, g_W2, g_b2])

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(all_params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Magic: 0x454D424C ("LBME" little-endian)
    # Header: magic, d, V, E, H, C (6 int32)
    magic = 0x454D424C
    header = struct.pack("<6i", magic, d, V, E, H, C)
    # Embeddings: d * V * E floats
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in emb)
    # W1, b1, W2, b2
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte class index. */
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], V = hd[2], E = hd[3], H = hd[4], C = hd[5];
    if (d != 16 || V != 256 || C < 1 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 24L + 4L * ((long)d * V * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *W1 = emb + (long)d * V * E;
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

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * V + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```