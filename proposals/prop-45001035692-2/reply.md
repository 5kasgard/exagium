```hypothesis
Family: label-embmlp – per-position byte embeddings (D=8) concatenated, one ReLU hidden layer (H=64), then linear to 65 classes. 
Why: The linear model (label-linear) uses 16 independent one-hot tables (4096×65 ≈ 266k params) but cannot capture cross-position interactions. A small embedding MLP reduces parameters to ~45k (16×256×8 + 128×64 + 64×65) while allowing non-linear feature combinations, which should improve accuracy on this character-level language task. Inference cost stays low because the embedding lookup is just 16 small table reads and the MLP is tiny.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear significantly, the added non-linearity does not help or the model is too small.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
D = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 256
EPOCHS = 20
SAMPLES = 600000

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
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
    d = X.shape[1]          # 16
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embedding tables: (d, 256, D)
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # Hidden layer: (d*D, H)
    W1 = (rng.normal(0, np.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer: (H, C)
    W2 = (rng.normal(0, np.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward: embedding lookup
            emb_batch = np.stack([emb[j, x_batch[:, j]] for j in range(d)], axis=1)  # (B, d, D)
            flat = emb_batch.reshape(B, d * D)                                      # (B, d*D)

            hidden_pre = flat @ W1 + b1                                             # (B, H)
            hidden = np.maximum(hidden_pre, 0)                                      # ReLU

            logits = hidden @ W2 + b2                                               # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = probs / B                                                 # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2                              # (H, C)
            grad_b2 = grad_logits.sum(axis=0)                                       # (C,)

            grad_hidden = grad_logits @ W2.T                                        # (B, H)
            grad_hidden[hidden_pre <= 0] = 0                                        # ReLU gradient

            grad_W1 = flat.T @ grad_hidden + L2 * W1                                # (d*D, H)
            grad_b1 = grad_hidden.sum(axis=0)                                       # (H,)

            grad_flat = grad_hidden @ W1.T                                          # (B, d*D)
            grad_emb = grad_flat.reshape(B, d, D)                                   # (B, d, D)

            emb_grad = np.zeros_like(emb)
            for j in range(d):
                np.add.at(emb_grad[j], x_batch[:, j], grad_emb[:, j, :])
            emb_grad += L2 * emb

            grads = [emb_grad, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Export weights
    magic = 0x454D4244  # "EMBD" in little-endian bytes
    out = struct.pack("<5i", magic, d, D, H, C)
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, D * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * D; idx++) {
                sum += flat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```