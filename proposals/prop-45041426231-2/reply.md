```hypothesis
Family: label-embed-mlp (per-position embeddings + one ReLU layer).  
Why: Character-level language modeling benefits from dense embeddings that capture lexical similarity (e.g., vowels near vowels) and a hidden layer adds non-linear context mixing. The 16-byte context is short, so a compact embedding (8 dims) and 64 hidden units keep inference fast (~10k Ir/item) while improving over linear (0.3375).  
Failure: If dev accuracy ≤ 0.3375 or instructions/item > 20000, the model does not justify its complexity.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, d, E, H, C; then embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
E = 8          # embedding dimension per byte position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
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
    # All input types are categorical -> use embeddings
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: list of d arrays, each (256, E)
    embeddings = [(rng.normal(0, 1/math.sqrt(E), (256, E))).astype(np.float32) for _ in range(d)]
    # Hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute split indices for each position
    pos_slices = [slice(j*E, (j+1)*E) for j in range(d)]

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            emb_concat = np.empty((B, d*E), np.float32)
            for j in range(d):
                emb_concat[:, pos_slices[j]] = embeddings[j][xb[:, j]]

            # Hidden layer
            hidden_pre = emb_concat @ W1 + b1          # (B, H)
            hidden = np.maximum(hidden_pre, 0)         # ReLU
            # Output
            logits = hidden @ W2 + b2                  # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                                 # (B, C) gradient w.r.t logits

            # Backprop output layer
            grad_W2 = hidden.T @ probs + L2 * W2       # (H, C)
            grad_b2 = probs.sum(axis=0)                # (C,)

            # Backprop hidden layer
            grad_hidden = probs @ W2.T                 # (B, H)
            grad_hidden[hidden_pre <= 0] = 0.0         # ReLU derivative

            # Backprop hidden weights and bias
            grad_W1 = emb_concat.T @ grad_hidden + L2 * W1   # (d*E, H)
            grad_b1 = grad_hidden.sum(axis=0)                # (H,)

            # Backprop embeddings
            grad_emb_concat = grad_hidden @ W1.T              # (B, d*E)
            grad_embeddings = []
            for j in range(d):
                g = grad_emb_concat[:, pos_slices[j]]        # (B, E)
                # Scatter-add into (256, E)
                ge = np.zeros((256, E), np.float32)
                np.add.at(ge, xb[:, j], g)
                grad_embeddings.append(ge)

            # Assemble gradients in same order as params
            grads = grad_embeddings + [grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C (5 int32)
    header = struct.pack("<5i", 0x454D4244, d, E, H, C)
    out = bytearray(header)
    # Embeddings
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    # W1, b1, W2, b2
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D4244, d, E, H, C (int32 LE), then
 *   embeddings[d][256][E] (float32), W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden_pre = (float *)malloc(sizeof(float) * H);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!hidden_pre || !hidden || !scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden_pre = b1
        memcpy(hidden_pre, b1, H * sizeof(float));
        // Accumulate embeddings * W1
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            const float *w1 = W1 + (long)j * E * H;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * w1[ee * H + h];
                hidden_pre[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) hidden[h] = hidden_pre[h] > 0.0f ? hidden_pre[h] : 0.0f;

        // Output scores = b2 + hidden * W2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w2 = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w2[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```