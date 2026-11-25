```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP).
Why: Character-level language modeling benefits from distributed representations; embeddings capture byte similarities and reduce parameter count vs one-hot linear. A small hidden layer adds non-linearity for context mixing. Expected to beat label-linear (0.3375) on dev accuracy with similar or lower CPU cost due to smaller input dimension (16*E vs 16*256).
Failure signal: dev accuracy ≤ 0.3375 or CPU cost per item significantly higher than label-linear.
```

```python
"""label-embmlp: Embedding + one-hidden-layer MLP for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8          # embedding dimension per byte position
H = 32         # hidden layer size
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    scale_emb = math.sqrt(3.0 / E)
    emb = rng.uniform(-scale_emb, scale_emb, (d, 256, E)).astype(np.float32)

    # Hidden layer: input size d*E, output H
    scale1 = math.sqrt(6.0 / (d * E + H))
    W1 = rng.uniform(-scale1, scale1, (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # Output layer: input H, output C
    scale2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale2, scale2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            # emb_x shape (B, d, E)
            emb_x = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, E)
            concat = emb_x.reshape(B, d * E)                    # (B, d*E)

            # Hidden layer
            hidden_pre = concat @ W1 + b1                        # (B, H)
            hidden = np.maximum(hidden_pre, 0)                   # ReLU

            # Output layer
            logits = hidden @ W2 + b2                            # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                                  # (B, C)

            # Gradients
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)

            dh = dlogits @ W2.T                                  # (B, H)
            dh[hidden_pre <= 0] = 0.0                            # ReLU gradient

            dW1 = concat.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)

            dconcat = dh @ W1.T                                  # (B, d*E)
            demb_x = dconcat.reshape(B, d, E)                    # (B, d, E)

            # Backprop to embeddings: demb has shape (d, 256, E)
            demb = np.zeros_like(emb)
            for j in range(d):
                # xb[:, j] are indices (B,)
                np.add.at(demb[j], xb[:, j], demb_x[:, j])

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights file
    # Header: magic, d, E, H, C (5 int32)
    header = struct.pack("<5i", 0x454D424C, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_flat = emb.transpose(0, 1, 2).reshape(-1).astype("<f4")
    # W1: (d*E, H) -> flatten row-major
    W1_flat = W1.reshape(-1).astype("<f4")
    b1_flat = b1.astype("<f4")
    W2_flat = W2.reshape(-1).astype("<f4")
    b2_flat = b2.astype("<f4")

    out = header + emb_flat.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (argmax).
 * Weights format: magic 0x454D424C, int32 d, E, H, C;
 * float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: concat @ W1 + b1, ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```