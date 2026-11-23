```hypothesis
We propose a factorized embedding MLP (label-embed-mlp) that shares token embeddings across the 16 positions and adds learned positional embeddings, followed by a single ReLU hidden layer. This captures positional interactions with far fewer parameters than a full per-position linear layer, which is crucial for the character-level language modeling structure of tinyshakespeare. The model should exceed label-linear's 33.75% accuracy while keeping inference cost competitive by using small embedding (8) and hidden (32) dimensions. If dev accuracy does not surpass label-linear or instruction count grows significantly, the factorization may be too restrictive for this task.
```
```python
"""Factorized embedding MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: shared token embedding (256 x E) + positional embedding (d x E) -> concatenate -> ReLU hidden (H) -> linear output (C).
Exported weights: magic, d, E, H, C; float32 arrays in order: E_tok[256*E], E_pos[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C]."""
import json, math, os, struct, sys

import numpy as np

# Hardened hyperparameters (chosen on tuning split)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
EMB_DIM = 8
HIDDEN = 32

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    E_tok = (rng.normal(0, 0.01, (256, EMB_DIM))).astype(np.float32)
    E_pos = (rng.normal(0, 0.01, (d, EMB_DIM))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * EMB_DIM)), (d * EMB_DIM, HIDDEN))).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / HIDDEN), (HIDDEN, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E_tok, E_pos, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = xb.shape[0]

            # Forward
            tok_emb = E_tok[xb]                     # (bs, d, E)
            pos_emb = E_pos[np.newaxis, :, :]       # (1, d, E)
            combined = tok_emb + pos_emb             # (bs, d, E)
            flat = combined.reshape(bs, d * EMB_DIM)
            hidden_pre = flat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            # Loss gradient
            grad_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            # Backward
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_flat = grad_hidden @ W1.T
            grad_combined = grad_flat.reshape(bs, d, EMB_DIM)

            grad_E_tok = np.zeros_like(E_tok)
            grad_E_pos = np.zeros_like(E_pos)
            for j in range(d):
                np.add.at(grad_E_tok, xb[:, j], grad_combined[:, j])
                grad_E_pos[j] = grad_combined[:, j].sum(axis=0)
            grad_E_tok += L2 * E_tok
            grad_E_pos += L2 * E_pos

            grads = [grad_E_tok, grad_E_pos, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x46424D45  # "EMBF"
    header = struct.pack("<5i", magic, d, EMB_DIM, HIDDEN, C)
    arrays = [E_tok, E_pos, W1, b1, W2, b2]
    body = b"".join(arr.astype("<f4").tobytes() for arr in arrays)
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* Factorized embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic (0x46424D45), d, E, H, C
 *   float32 E_tok[256*E], E_pos[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x46424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected = 20
        + 4L * (256L * E + (long)d * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E_tok = (const float *)(hd + 5);
    const float *E_pos = E_tok + 256L * E;
    const float *W1 = E_pos + (long)d * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *combined = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build combined embeddings
        for (int j = 0; j < d; j++) {
            const float *tok = E_tok + (long)xi[j] * E;
            const float *pos_emb = E_pos + (long)j * E;
            float *out = combined + (long)j * E;
            for (int e = 0; e < E; e++) out[e] = tok[e] + pos_emb[e];
        }

        // Hidden layer: ReLU(combined * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += combined[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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