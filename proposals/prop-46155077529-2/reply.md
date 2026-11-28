```hypothesis
We propose "label-emb-linear": a model that learns a dense embedding for each of the 16 byte positions (256 vocab, embedding dim E=8) followed by a linear classifier. This captures character similarities (e.g., letters vs punctuation) that one-hot encoding ignores, potentially improving accuracy over label-linear (0.3375) while keeping inference fast (only 16*E*C multiply-adds per item). If dev accuracy does not exceed 0.3375, the hypothesis fails.
```

```python
"""label-emb-linear: embeddings + linear classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8            # embedding dimension per position
SEED = 1
LR = 0.01
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    # embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # weight matrix: (d*E, C)
    W = rng.normal(0, math.sqrt(2.0 / (d*E + C)), size=(d*E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute one-hot for y
    y_onehot = np.zeros((n, C), dtype=np.float32)
    y_onehot[np.arange(n), y] = 1.0

    # Training loop
    max_steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]          # (B, d)
            yb = y_onehot[idx]   # (B, C)
            B = Xb.shape[0]

            # Forward: build hidden vector (B, d*E)
            hidden = np.zeros((B, d*E), dtype=np.float32)
            for j in range(d):
                # emb[Xb[:, j]] -> (B, E)
                hidden[:, j*E:(j+1)*E] = emb[j, Xb[:, j]]

            logits = hidden @ W + b                    # (B, C)
            # Softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Loss gradient: dlogits = (probs - yb) / B
            dlogits = (probs - yb) / B                # (B, C)

            # Gradients
            dW = hidden.T @ dlogits + L2 * W          # (d*E, C)
            db = dlogits.sum(axis=0)                  # (C,)
            dhidden = dlogits @ W.T                   # (B, d*E)

            # Gradient for embeddings
            demb = np.zeros_like(emb)
            for j in range(d):
                # dhidden[:, j*E:(j+1)*E] is (B, E)
                # Scatter-add to demb[j, Xb[:, j], :]
                # We can use a loop over batch (B is small) or np.add.at
                for i in range(B):
                    demb[j, Xb[i, j]] += dhidden[i, j*E:(j+1)*E]

            grads = [demb, dW, db]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Magic: 0x454D424C ('LBME' little-endian)
    magic = 0x454D424C
    header = struct.pack("<4i", magic, d, E, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # Weight matrix: (d*E) * C floats
    W_bytes = W.astype("<f4").tobytes()
    # Bias: C floats
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D424C), d, E, C (int32), then embeddings[d][256][E], W[d*E][C], b[C] as float32. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * C + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build hidden vector
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(hidden + j * E, e, E * sizeof(float));
        }

        // Compute scores = hidden @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k_idx = 0; k_idx < d * E; k_idx++) {
            float h = hidden[k_idx];
            const float *w = W + (long)k_idx * C;
            for (int c = 0; c < C; c++) scores[c] += h * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden);
    free(scores);
    return 0;
}
```