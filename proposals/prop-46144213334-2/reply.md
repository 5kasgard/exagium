```hypothesis
We propose "label-embmlp": a small MLP with per-position byte embeddings. The 16-byte context is embedded into 16 vectors of size E, concatenated, passed through a ReLU hidden layer of size H, then a linear output layer. Embeddings reduce the input dimension from 4096 (one-hot) to 16*E, allowing a larger hidden layer for similar compute. On tinyshakespeare, local character patterns benefit from learned embeddings. If dev accuracy does not exceed label-linear (0.3375) at comparable or lower CPU cost, the hypothesis fails.
```

```python
"""Embedding MLP for next-character prediction: 16 categorical bytes -> embeddings -> ReLU hidden -> linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 32        # embedding size per position
H = 128       # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50

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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: (d*E) x H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H x C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            # Embedding lookup: (bs, d, E)
            emb_lookup = emb[np.arange(d)[None, :, None], xb[:, :, None], np.arange(E)[None, None, :]]  # (bs, d, E)
            cat = emb_lookup.reshape(bs, d * E)  # (bs, d*E)
            hidden_pre = cat @ W1 + b1  # (bs, H)
            hidden = np.maximum(hidden_pre, 0)  # ReLU
            logits = hidden @ W2 + b2  # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs  # (bs, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T  # (bs, H)
            grad_hidden[hidden_pre <= 0] = 0.0  # ReLU derivative
            grad_W1 = cat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            grad_cat = grad_hidden @ W1.T  # (bs, d*E)
            grad_emb_lookup = grad_cat.reshape(bs, d, E)
            # Efficient scatter-add using np.add.at
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_lookup[:, j, :])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C
    magic = 0x454D424D  # "EMBM"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten embeddings in order: position 0..d-1, each 256 vectors of E floats
    emb_flat = emb.transpose(0, 1, 2).reshape(-1).astype("<f4")
    W1_flat = W1.astype("<f4").ravel()
    b1_flat = b1.astype("<f4")
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4")
    out = header + emb_flat.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding MLP inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20
        + (long)d * 256 * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *cat = malloc((long)d * E * sizeof(float));
    float *hidden = malloc((long)H * sizeof(float));
    float *logits = malloc((long)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(cat + (long)j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(cat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int idx = 0; idx < d * E; idx++) acc += cat[idx] * W1[(long)idx * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
            logits[c] = acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(cat); free(hidden); free(logits);
    if (x) free(x);
    free(hd);
    return 0;
}
```