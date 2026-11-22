```hypothesis
Family: label-emb (embedding + 1-hidden-layer MLP). Each of the 16 context positions gets a learned 8-d embedding, concatenated (128-d), then a ReLU layer of 64 units, then 65-class output. This captures byte similarities (e.g., letters near each other) with far fewer parameters than label-linear's one-hot (45k vs 266k), so it should train faster and generalize better on this character-level task. If dev accuracy ≤ 0.3375 or CPU cost ≥ label-linear, the hypothesis fails.
```
```python
"""label-emb: embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x4C424D45, d, E, H, C; float32 embed[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
SEED = 1
EMB_DIM = 8        # E
HIDDEN = 64        # H
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 10
SAMPLES = 600000   # max samples per epoch budget

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    scale_emb = math.sqrt(6.0 / 256)
    embed = (rng.uniform(-1, 1, (d, 256, EMB_DIM)) * scale_emb).astype(np.float32)

    # First layer: (d*E, H)
    in_dim = d * EMB_DIM
    scale1 = math.sqrt(6.0 / in_dim)
    W1 = (rng.uniform(-1, 1, (in_dim, HIDDEN)) * scale1).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)

    # Second layer: (H, C)
    scale2 = math.sqrt(6.0 / HIDDEN)
    W2 = (rng.uniform(-1, 1, (HIDDEN, C)) * scale2).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    # Adam state
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

            # Forward
            # Embedding lookup: (B, d, E)
            emb_out = embed[np.arange(d)[None, :], xb]   # (B, d, E)
            flat = emb_out.reshape(B, -1)                # (B, d*E)
            h = flat @ W1 + b1                           # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2                    # (B, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            # grad_W2 = h_relu.T @ grad_logits
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            grad_h_relu = grad_logits @ W2.T               # (B, H)
            grad_h = grad_h_relu * (h > 0)                 # ReLU grad

            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            # Embedding gradients: (B, d, E) from grad_h reshaped
            grad_flat = grad_h @ W1.T                      # (B, d*E)
            grad_emb = grad_flat.reshape(B, d, EMB_DIM)    # (B, d, E)
            # Accumulate per (position, byte)
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                bytes_j = xb[:, j]                         # (B,)
                # scatter add
                np.add.at(grad_embed[j], bytes_j, grad_emb[:, j, :])
            grad_embed += L2 * embed

            grads = [grad_embed, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, H, C
    header = struct.pack("<5i", 0x4C424D45, d, EMB_DIM, HIDDEN, C)
    # embeddings: (d, 256, E) -> flatten to (d*256*E)
    emb_bytes = embed.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x4C424D45, int32 d, E, H, C;
 * float32 embed[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    float *emb_concat = (float *)malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + (long)j * E, e, sizeof(float) * E);
        }

        // First layer: emb_concat (d*E) @ W1 (d*E x H) + b1
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) {
                sum += emb_concat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }

        // Second layer: hidden (H) @ W2 (H x C) + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores); free(emb_concat);
    return 0;
}
```