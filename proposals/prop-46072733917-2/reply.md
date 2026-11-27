```hypothesis
Family: label-embed-mlp – shared byte embeddings (256×8) + one ReLU layer (64 units) + linear output.
Why: One-hot linear models ignore byte similarities; embeddings generalize across 256 values and positions, while a small MLP captures cross-position interactions. This should beat label-linear's 33.75% accuracy with fewer parameters and lower inference cost.
Failure: If dev accuracy ≤ 0.3375 or Ir/item ≥ 7359 (label-linear), the embedding+MLP adds complexity without benefit.
```

```python
"""label-embed-mlp: shared embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, int32 d,emb_dim,H,C; float32 emb[256*emb_dim], W1[(d*emb_dim)*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
EMB_DIM = 8
H = 64
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Xavier init for embeddings
    scale_emb = math.sqrt(6.0 / (256 + EMB_DIM))
    emb = (rng.uniform(-1, 1, (256, EMB_DIM)) * scale_emb).astype(np.float32)
    # W1: (d*EMB_DIM) x H
    k = d * EMB_DIM
    scale1 = math.sqrt(6.0 / (k + H))
    W1 = (rng.uniform(-1, 1, (k, H)) * scale1).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C
    scale2 = math.sqrt(6.0 / (H + C))
    W2 = (rng.uniform(-1, 1, (H, C)) * scale2).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Embedding lookup: (B, d, EMB_DIM)
            x_emb = emb[xb]      # advanced indexing
            x_flat = x_emb.reshape(B, k)  # (B, k)

            # Forward
            hidden = x_flat @ W1 + b1          # (B, H)
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2     # (B, C)

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                 # (B, C)

            # Gradients
            dW2 = hidden_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden_relu = dlogits @ W2.T
            dhidden = dhidden_relu * (hidden > 0)   # ReLU grad
            dW1 = x_flat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dx_flat = dhidden @ W1.T                # (B, k)
            dx_emb = dx_flat.reshape(B, d, EMB_DIM) # (B, d, EMB_DIM)

            # Accumulate embedding gradients
            demb = np.zeros_like(emb)
            for pos in range(d):
                np.add.at(demb, xb[:, pos], dx_emb[:, pos, :])
            demb += L2 * emb

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424C  # 'LMBE' in little-endian
    header = struct.pack("<5i", magic, d, EMB_DIM, H, C)
    out = header
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || emb_dim <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    int k = d * emb_dim;
    long expected = 20L + 4L * (256L * emb_dim + (long)k * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * emb_dim;
    const float *b1 = W1 + (long)k * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Concatenate embeddings
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * emb_dim;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e_idx = 0; e_idx < emb_dim; e_idx++) {
                    sum += e[e_idx] * W1[(j * emb_dim + e_idx) * H + h];
                }
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w2 = W2 + (long)h * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w2[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```