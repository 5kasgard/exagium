```hypothesis
Family: label-emb-mlp (factorized embedding MLP). Each of the 16 context bytes gets a learned embedding (size E), concatenated, then a ReLU hidden layer (size H), then linear output to 65 classes. This captures cross-position interactions and shares statistical strength across byte values, which a linear per-position model cannot. Factorizing the first layer (embeddings + hidden weights) reduces parameters vs. a full position-byte table, improving generalization. Expect dev accuracy >0.34 with CPU cost comparable to label-linear. Failure if dev accuracy ≤0.3375 or instructions/item exceed label-linear by >20%.
```

```python
"""label-emb-mlp: factorized embedding MLP for fixed-size byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position embeddings (d x 256 x E) -> concat -> ReLU hidden (H) -> linear output (C).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  magic(0x454D424C), d, E, H, C (int32 each)
  embeddings[d][256][E] (float32)
  hidden_weight[(d*E)][H] (float32)
  hidden_bias[H] (float32)
  output_weight[H][C] (float32)
  output_bias[C] (float32)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters
E = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424C  # 'LMBE' in little-endian ASCII


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
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # hidden: (d*E, H)
    W_hidden = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b_hidden = np.zeros(H, dtype=np.float32)
    # output: (H, C)
    W_out = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [emb, W_hidden, b_hidden, W_out, b_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute one-hot for y
    y_onehot = np.zeros((n, C), dtype=np.float32)
    y_onehot[np.arange(n), y] = 1.0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]          # (B, d)
            yb = y_onehot[idx]         # (B, C)
            B = len(idx)

            # Forward
            # Embedding lookup: (B, d, E)
            emb_seq = emb[np.arange(d), Xb.T].transpose(1, 0, 2)  # (B, d, E)
            emb_flat = emb_seq.reshape(B, d * E)                  # (B, d*E)
            h_pre = emb_flat @ W_hidden + b_hidden                # (B, H)
            h = np.maximum(h_pre, 0)                              # ReLU
            logits = h @ W_out + b_out                            # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = (probs - yb) / B                            # (B, C)

            # Backprop output
            dW_out = h.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(axis=0)
            dh = dlogits @ W_out.T                                # (B, H)

            # Backprop hidden
            dh_pre = dh * (h_pre > 0)                             # (B, H)
            dW_hidden = emb_flat.T @ dh_pre + L2 * W_hidden
            db_hidden = dh_pre.sum(axis=0)
            demb_flat = dh_pre @ W_hidden.T                       # (B, d*E)
            demb_seq = demb_flat.reshape(B, d, E)                 # (B, d, E)

            # Backprop embeddings: scatter-add
            demb = np.zeros_like(emb)
            # For each position j, accumulate gradients for the byte values in the batch
            # Vectorized: use np.add.at on the second axis (256)
            for j in range(d):
                np.add.at(demb[j], Xb[:, j], demb_seq[:, j])

            grads = [demb, dW_hidden, db_hidden, dW_out, db_out]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Order: embeddings (d,256,E), W_hidden (d*E,H), b_hidden (H), W_out (H,C), b_out (C)
    out_bytes = struct.pack("<5i", MAGIC, d, E, H, C)
    out_bytes += emb.astype("<f4").tobytes()
    out_bytes += W_hidden.astype("<f4").tobytes()
    out_bytes += b_hidden.astype("<f4").tobytes()
    out_bytes += W_out.astype("<f4").tobytes()
    out_bytes += b_out.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format (little-endian):
 *   magic(0x454D424C), d, E, H, C (int32)
 *   embeddings[d][256][E] (float32)
 *   hidden_weight[(d*E)][H] (float32)
 *   hidden_bias[H] (float32)
 *   output_weight[H][C] (float32)
 *   output_bias[C] (float32)
 */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size
    long expected = 20
        + (long)d * 256 * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W_hidden = emb + (long)d * 256 * E;
    const float *b_hidden = W_hidden + (long)d * E * H;
    const float *W_out = b_hidden + H;
    const float *b_out = W_out + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers
    float *emb_flat = (float *)malloc((size_t)d * E * sizeof(float));
    float *hidden = (float *)malloc((size_t)H * sizeof(float));
    float *logits = (float *)malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: h = ReLU(emb_flat * W_hidden + b_hidden)
        for (int h = 0; h < H; h++) {
            float sum = b_hidden[h];
            for (int idx = 0; idx < d * E; idx++) {
                sum += emb_flat[idx] * W_hidden[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = hidden * W_out + b_out
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W_out[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(emb_flat);
    free(hidden);
    free(logits);
    if (x) free(x);
    free(hd);
    return 0;
}
```