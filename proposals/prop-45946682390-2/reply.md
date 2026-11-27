```hypothesis
Family: embedding-mlp (position-wise embeddings summed, one ReLU hidden layer).
Why: The 16 context bytes are categorical; one-hot linear uses 4096 inputs. Learned embeddings (E=32) reduce dimensionality, capture character similarities, and the hidden layer (H=64) models interactions. This should exceed label-linear's 0.3375 accuracy with fewer parameters (~137k vs 266k) and lower inference cost.
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear's 7359.9, the hypothesis is disproved.
```

```python
"""Embedding-MLP family for fixed-size byte items: position-wise embeddings summed, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
E = 32          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424D  # 'MBME' unique magic

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
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)      # (d, 256, E)
    W1  = rng.normal(0, math.sqrt(2.0 / E), (E, H)).astype(np.float32)
    b1  = np.zeros(H, np.float32)
    W2  = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2  = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state  = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    # Precompute flat embedding matrix for fast gather
    flat_emb_shape = (d * 256, E)
    pos_offsets = np.arange(d) * 256

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]                 # (B, d)
            y_batch = y[idx]                 # (B,)
            B = len(idx)

            # Embedding lookup and sum
            flat_emb = emb.reshape(flat_emb_shape)                     # (d*256, E)
            indices = x_batch + pos_offsets                            # (B, d)
            emb_seqs = flat_emb[indices]                               # (B, d, E)
            emb_sum = emb_seqs.sum(axis=1)                             # (B, E)

            # Forward pass
            hidden_pre = emb_sum @ W1 + b1                             # (B, H)
            hidden = np.maximum(hidden_pre, 0)                         # ReLU
            logits = hidden @ W2 + b2                                  # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), y_batch] -= 1.0
            grad_logits /= B                                           # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden_relu = grad_logits @ W2.T
            grad_hidden_pre = grad_hidden_relu * (hidden_pre > 0)
            grad_W1 = emb_sum.T @ grad_hidden_pre + L2 * W1
            grad_b1 = grad_hidden_pre.sum(axis=0)
            grad_emb_sum = grad_hidden_pre @ W1.T                      # (B, E)

            # Gradient for embeddings (scatter-add per position)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], x_batch[:, j], grad_emb_sum)
            grad_emb += L2 * emb

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

    # Export weights
    out = struct.pack("<5i", MAGIC, d, E, H, C)
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
/* Embedding-MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | (buf[*pos+1] << 8) | (buf[*pos+2] << 16) | ((uint32_t)buf[*pos+3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long wlen, xlen = 0;
    long N = atol(argv[3]);
    int32_t *hdr = (int32_t *)slurp(argv[1], &wlen);
    if (wlen < 20 || hdr[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], E = hdr[2], H = hdr[3], C = hdr[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }

    size_t expected = 20
        + (size_t)d * 256 * E * 4
        + (size_t)E * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (wlen != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hdr + 5);
    const float *W1  = emb + (size_t)d * 256 * E;
    const float *b1  = W1  + (size_t)E * H;
    const float *W2  = b1  + H;
    const float *b2  = W2  + (size_t)H * C;

    unsigned char *xbuf = N ? slurp(argv[2], &xlen) : NULL;
    long pos = 0;

    float *emb_sum = (float *)malloc(sizeof(float) * E);
    float *hidden  = (float *)malloc(sizeof(float) * H);
    float *output  = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(xbuf, xlen, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero emb_sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // Sum embeddings for each position
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // Hidden layer: hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int e = 0; e < E; e++) {
            if (emb_sum[e] == 0.0f) continue;
            const float *w1_row = W1 + (size_t)e * H;
            for (int h = 0; h < H; h++) hidden[h] += emb_sum[e] * w1_row[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer: output = hidden @ W2 + b2
        for (int c = 0; c < C; c++) output[c] = b2[c];
        for (int h = 0; h < H; h++) {
            if (hidden[h] == 0.0f) continue;
            const float *w2_row = W2 + (size_t)h * C;
            for (int c = 0; c < C; c++) output[c] += hidden[h] * w2_row[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(output);
    if (xbuf) free(xbuf);
    free(hdr);
    return 0;
}
```