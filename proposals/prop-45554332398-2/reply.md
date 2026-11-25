```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP).
Why: The task is next-character prediction from 16 bytes of context. A linear model on one-hot features cannot capture interactions between positions and characters. Using learned embeddings per position (8-d) followed by a ReLU hidden layer (64 units) allows the model to represent compositional features and should exceed label-linear's 0.3375 accuracy. The model is still small (~45k parameters) and inference is a few matrix-vector products, keeping CPU cost low.
Failure signal: If dev accuracy stays ≤0.3375 or inference cost exceeds label-linear by >2×, the family does not meet its goal.
```

```python
"""label-embmlp: embedding + one-hidden-layer MLP for fixed-size byte sequence classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, int32 d,E,H,C; float32 embeddings[d][256][E], W_hid[d*E][H], b_hid[H], W_out[H][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
MAGIC = 0x454D4244  # 'EMBD' in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: list of d arrays, each (256, E)
    embeddings = [rng.normal(0, 0.01, size=(256, E)).astype(np.float32) for _ in range(d)]
    W_hid = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b_hid = np.zeros(H, dtype=np.float32)
    W_out = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = embeddings + [W_hid, b_hid, W_out, b_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: for each position j, get (B, E)
            emb_out = np.empty((B, d * E), dtype=np.float32)
            for j in range(d):
                emb_out[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]
            hid_pre = emb_out @ W_hid + b_hid          # (B, H)
            hid = np.maximum(hid_pre, 0)               # ReLU
            logits = hid @ W_out + b_out               # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                        # (B, C)

            # Backprop output layer
            dW_out = hid.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(axis=0)
            dhid = dlogits @ W_out.T                   # (B, H)
            dhid[hid_pre <= 0] = 0                     # ReLU gradient

            # Backprop hidden layer
            dW_hid = emb_out.T @ dhid + L2 * W_hid
            db_hid = dhid.sum(axis=0)
            demb_out = dhid @ W_hid.T                  # (B, d*E)

            # Backprop embeddings
            demb = [np.zeros_like(emb) for emb in embeddings]
            for j in range(d):
                # demb_out[:, j*E:(j+1)*E] gradients for position j
                # Accumulate using np.add.at for the used indices
                np.add.at(demb[j], xb[:, j], demb_out[:, j*E:(j+1)*E])

            # Collect gradients in same order as params
            grads = demb + [dW_hid, db_hid, dW_out, db_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C (all int32)
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = b"".join(emb.astype("<f4").tobytes() for emb in embeddings)
    # Hidden weights and bias
    hid_bytes = W_hid.astype("<f4").tobytes() + b_hid.astype("<f4").tobytes()
    # Output weights and bias
    out_bytes = W_out.astype("<f4").tobytes() + b_out.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header + emb_bytes + hid_bytes + out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D4244, int32 d,E,H,C; float32 embeddings[d][256][E],
 *   W_hid[d*E][H], b_hid[H], W_out[H][C], b_out[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Expected size: header(20) + embeddings(d*256*E) + W_hid(d*E*H) + b_hid(H) + W_out(H*C) + b_out(C)
    long expected = 20 + 4L * ( (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W_hid = emb + (long)d * 256 * E;
    const float *b_hid = W_hid + (long)d * E * H;
    const float *W_out = b_hid + H;
    const float *b_out = W_out + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers for inference (max sizes: d*E <= 16*16=256, H <= 256, C=65)
    float input_vec[256];
    float hidden[256];
    float scores[65];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector by concatenating embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(input_vec + j * E, e, E * sizeof(float));
        }

        // Hidden layer: input_vec (d*E) * W_hid (d*E x H) + b_hid -> ReLU
        for (int h = 0; h < H; h++) {
            float sum = b_hid[h];
            for (int idx = 0; idx < d * E; idx++) {
                sum += input_vec[idx] * W_hid[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden (H) * W_out (H x C) + b_out
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W_out[(long)h * C + c];
            }
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```