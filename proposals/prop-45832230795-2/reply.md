```hypothesis
Family: label-embmlp (per-position byte embeddings + one ReLU hidden layer).
Why: The linear model uses 16×256 one-hot features (4096 dim) directly to 65 classes. Replacing one-hot with learned embeddings (E=32) reduces input dimension to 512, adds a 128-unit ReLU layer to capture pairwise position interactions, and cuts multiply-adds from ~266k to ~74k per item. This should raise accuracy above label-linear's 0.3375 while being faster.
Failure signal: dev accuracy ≤ 0.340 or inference cost ≥ label-linear.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 256
EPOCHS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]   # 16
    C = card["classes"]          # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # Hidden weights: (d*E, H)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output weights: (H, C)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Embedding lookup: (B, d, E)
            emb_batch = emb[np.arange(d), xb.T].transpose(1, 0, 2)
            flat = emb_batch.reshape(B, -1)                 # (B, d*E)

            # Forward
            h = flat @ W1 + b1                              # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2                       # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W2 = h_relu.T @ grad_logits                # (H, C)
            grad_b2 = grad_logits.sum(axis=0)               # (C,)
            grad_h_relu = grad_logits @ W2.T                # (B, H)
            grad_h = grad_h_relu * (h > 0)                  # ReLU grad
            grad_W1 = flat.T @ grad_h                       # (d*E, H)
            grad_b1 = grad_h.sum(axis=0)                    # (H,)
            grad_flat = grad_h @ W1.T                       # (B, d*E)

            # Embedding gradients
            grad_emb_batch = grad_flat.reshape(B, d, E)     # (B, d, E)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_batch[:, j])

            # L2
            grad_W1 += L2 * W1
            grad_b1 += L2 * b1
            grad_W2 += L2 * W2
            grad_b2 += L2 * b2
            grad_emb += L2 * emb

            # Adam update
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights: magic, d, E, H, C, then arrays in order
    magic = 0x454D424D
    out = struct.pack("<5i", magic, d, E, H, C)
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1  = emb + (long)d * 256 * E;
    const float *b1  = W1  + (long)d * E * H;
    const float *W2  = b1  + H;
    const float *b2  = W2  + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(hidden, b1, H * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *e = emb + ((long)j * 256 + byte) * E;
            const float *W1_j = W1 + (long)j * E * H;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e_idx = 0; e_idx < E; e_idx++) {
                    sum += e[e_idx] * W1_j[(long)e_idx * H + h];
                }
                hidden[h] += sum;
            }
        }

        for (int h = 0; h < H; h++)
            if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++)
                sum += hidden[h] * W2[(long)h * C + c];
            output[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden);
    free(output);
    return 0;
}
```