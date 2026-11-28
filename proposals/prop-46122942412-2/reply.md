```hypothesis
Family: label-embed (embedding + one-hidden-layer MLP). Each of the 16 context positions gets a learned 8-dim embedding per byte value; concatenated embeddings (128-d) feed a ReLU hidden layer (64 units) then a 65-class linear head. This captures cross-position interactions that the linear model misses, while keeping inference fast (≈12k FLOPs/item vs linear's ≈266k). Expect dev accuracy >0.35 with lower CPU cost than label-linear. Failure: accuracy ≤0.34 or Ir/item >8000.
```
```python
"""label-embed: embedding + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only.
Weights format: magic 0x454D4244, d, D, H, C; float32 embeddings[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 8          # embedding dimension per position
H = 64         # hidden layer size
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
    # embeddings: (d, 256, D)
    scale_emb = math.sqrt(6.0 / (256 + D))
    embeddings = rng.uniform(-scale_emb, scale_emb, (d, 256, D)).astype(np.float32)

    # W1: (d*D, H), b1: (H,)
    in_dim = d * D
    scale1 = math.sqrt(6.0 / (in_dim + H))
    W1 = rng.uniform(-scale1, scale1, (in_dim, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # W2: (H, C), b2: (C,)
    scale2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale2, scale2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: embeddings lookup
            # emb_concat: (B, d*D)
            emb_concat = np.zeros((B, in_dim), np.float32)
            for j in range(d):
                emb_concat[:, j*D:(j+1)*D] = embeddings[j, xb[:, j]]

            # Hidden layer
            hidden_pre = emb_concat @ W1 + b1          # (B, H)
            hidden = np.maximum(hidden_pre, 0)         # ReLU
            logits = hidden @ W2 + b2                  # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T               # (B, H)
            grad_hidden[hidden_pre <= 0] = 0.0             # ReLU grad

            grad_W1 = emb_concat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb_concat = grad_hidden @ W1.T           # (B, d*D)

            # Gradient for embeddings
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                # grad_emb_concat[:, j*D:(j+1)*D] corresponds to position j
                g = grad_emb_concat[:, j*D:(j+1)*D]
                # Accumulate using advanced indexing
                np.add.at(grad_embeddings[j], xb[:, j], g)

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
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
    # magic, d, D, H, C
    header = struct.pack("<5i", 0x454D4244, d, D, H, C)
    # embeddings (d, 256, D) -> flat
    emb_bytes = embeddings.astype("<f4").tobytes()
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long in_dim = (long)d * D;
    long expected_floats = (long)d * 256 * D + in_dim * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + in_dim * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * in_dim);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(emb_concat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int m = 0; m < in_dim; m++) sum += emb_concat[m] * W1[(long)m * H + h];
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