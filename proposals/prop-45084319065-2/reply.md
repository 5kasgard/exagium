```hypothesis
We propose a "label-embed" family: position-specific embeddings for the 65-character vocabulary followed by a linear classifier. This factorizes the large one-hot weight matrix (16*256*65) into a small embedding table (16*66*8) and a linear layer (128*65), reducing parameters from 266k to ~17k. The embeddings capture character similarities (e.g., letters cluster together) and should generalize better on the small tinyshakespeare dataset. If dev accuracy does not exceed label-linear's 0.3375, the factorization does not help for this task.
```

```python
"""label-embed: position-specific embeddings + linear classifier for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8                # embedding size
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 256
EPOCHS = 20
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X_bytes.shape
    C = card["classes"]
    # Vocabulary from class_chars (order matters)
    vocab_chars = [c.encode('ascii')[0] for c in card["class_chars"]]
    V = len(vocab_chars)  # 65
    # Mapping byte value -> index (0..V-1), unknown -> V
    byte_to_idx = np.full(256, V, dtype=np.int32)
    for idx, bval in enumerate(vocab_chars):
        byte_to_idx[bval] = idx
    X_idx = byte_to_idx[X_bytes]  # (n, d) int32

    # Initialize parameters
    rng = np.random.default_rng(SEED)
    # Embeddings: (d, V+1, E)
    emb = rng.normal(0, 0.01, size=(d, V+1, E)).astype(np.float32)
    # Linear weights: (d*E, C)
    W = rng.normal(0, math.sqrt(2.0 / (d*E)), size=(d*E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    params = [emb, W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_idx = X_idx[idx]          # (B, d)
            yy = y[idx]                 # (B,)
            B = len(idx)

            # Forward: gather embeddings and concatenate
            # emb[x_idx] has shape (B, d, E) -> reshape to (B, d*E)
            x_emb = emb[np.arange(d)[None, :], x_idx].reshape(B, d*E)  # (B, d*E)
            logits = x_emb @ W + b        # (B, C)
            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient of loss w.r.t logits
            grad_logits = probs
            grad_logits[np.arange(B), yy] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W = x_emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_x_emb = grad_logits @ W.T   # (B, d*E)
            grad_x_emb = grad_x_emb.reshape(B, d, E)  # (B, d, E)

            # Gradient for embeddings: scatter add
            grad_emb = np.zeros_like(emb)
            # For each position p, we have indices x_idx[:, p] and grad_x_emb[:, p, :]
            for p in range(d):
                np.add.at(grad_emb[p], x_idx[:, p], grad_x_emb[:, p, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * gr
                v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * (gr * gr)
                m_hat = m[i] / (1 - ADAM_BETA1 ** t)
                v_hat = v[i] / (1 - ADAM_BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    # Export weights
    # Format: magic (uint32), d, E, C, V (each uint32), then V bytes vocab,
    # then embeddings (d*(V+1)*E float32), then W (d*E*C float32), then b (C float32)
    magic = 0x44424D45  # 'EMBD' little-endian
    header = struct.pack("<5I", magic, d, E, C, V)
    vocab_bytes = bytes(vocab_chars)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + vocab_bytes + emb_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index (0..C-1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t len) {
    unsigned char hdr[4] = {len & 255, (len>>8)&255, (len>>16)&255, (len>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, len, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    unsigned char *wbuf = slurp(argv[1], &nw);
    unsigned char *xbuf = N ? slurp(argv[2], &nx) : NULL;

    // Parse header
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wbuf;
    if (hdr[0] != 0x44424D45) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1];
    int E = hdr[2];
    int C = hdr[3];
    int V = hdr[4];
    if (d != 16 || C != 65 || V != 65 || E != 8) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    // Vocabulary
    const unsigned char *vocab = wbuf + 20;
    // Build lookup table: byte value -> index (0..V), unknown -> V
    uint8_t lut[256];
    for (int i = 0; i < 256; i++) lut[i] = V;
    for (int i = 0; i < V; i++) lut[vocab[i]] = i;

    // Pointers to arrays
    const float *emb = (const float *)(vocab + V);
    const float *W = emb + (long)d * (V + 1) * E;
    const float *b = W + (long)d * E * C;

    // Validate total size
    long expected = 20 + V
                  + (long)d * (V + 1) * E * 4
                  + (long)d * E * C * 4
                  + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    // Temporary buffers
    float *x_emb = malloc((long)d * E * sizeof(float));
    float *scores = malloc((long)C * sizeof(float));
    long pos = 0;

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *rec = next_rec(xbuf, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings
        for (int p = 0; p < d; p++) {
            int idx = lut[rec[p]];
            const float *e = emb + ((long)p * (V + 1) + idx) * E;
            memcpy(x_emb + p * E, e, E * sizeof(float));
        }

        // Linear layer: scores = x_emb @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * E; k++) {
            float xv = x_emb[k];
            if (xv == 0.0f) continue;
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += xv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_emb);
    free(scores);
    return 0;
}
```