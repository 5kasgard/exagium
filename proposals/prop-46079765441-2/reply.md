```hypothesis
Family: label-factorized-mlp. A one-hidden-layer MLP where the first-layer weights are factorized into a shared byte embedding (256×D) and a position embedding (16×D), summed across the 16 context positions, then passed through a ReLU hidden layer of size H and a linear output layer. This reduces parameters vs. a full one-hot linear layer (label-linear) while capturing positional context and non-linear interactions. Expected to beat label-linear (33.75% dev accuracy) because the factorization acts as a strong inductive bias for character-level sequences. If it fails to exceed label-linear, it would indicate the factorization is too restrictive for this task.
```
```python
"""Factorized MLP for next-byte prediction: shared byte embeddings + position embeddings -> ReLU hidden -> output.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x46414354, d, D, H, C; float32 byte_emb[256][D], pos_emb[d][D], W1[D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen a priori; could be tuned on last 20% of train.in)
D = 128          # embedding dimension
H = 128          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
EPOCHS = 50
MAGIC = 0x46414354  # "FACT" in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def adam_update(param, grad, m, v, t, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    m *= beta1; m += (1 - beta1) * grad
    v *= beta2; v += (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    byte_emb = rng.normal(0, 0.01, (256, D)).astype(np.float32)
    pos_emb  = rng.normal(0, 0.01, (d, D)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / D), (D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [byte_emb, pos_emb, W1, b1, W2, b2]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: sum of byte_emb + pos_emb over positions
            emb_sum = np.zeros((B, D), np.float32)
            for j in range(d):
                emb_sum += byte_emb[xb[:, j]] + pos_emb[j]

            hidden = np.maximum(emb_sum @ W1 + b1, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden == 0] = 0.0
            grad_W1 = emb_sum.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb_sum = grad_hidden @ W1.T

            # Gradients for embeddings
            grad_byte_emb = np.zeros_like(byte_emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            flat_x = xb.ravel()
            flat_grad = np.repeat(grad_emb_sum, d, axis=0)
            np.add.at(grad_byte_emb, flat_x, flat_grad)
            for j in range(d):
                grad_pos_emb[j] += grad_emb_sum.sum(axis=0)

            # L2 regularization
            grad_W2 += L2 * W2
            grad_W1 += L2 * W1
            grad_byte_emb += L2 * byte_emb
            grad_pos_emb += L2 * pos_emb

            grads = [grad_byte_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for p, g, (m, v) in zip(params, grads, states):
                adam_update(p, g, m, v, t)

    # Pack weights.bin
    header = struct.pack("<5i", MAGIC, d, D, H, C)
    out = bytearray()
    out += header
    out += byte_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorized MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x46414354, d, D, H, C; float32 byte_emb[256][D], pos_emb[d][D], W1[D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * (256L * D + (long)d * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *byte_emb = (const float *)(hd + 5);
    const float *pos_emb  = byte_emb + 256L * D;
    const float *W1       = pos_emb  + (long)d * D;
    const float *b1       = W1       + (long)D * H;
    const float *W2       = b1       + H;
    const float *b2       = W2       + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *hidden  = malloc(sizeof(float) * H);
    float *logits  = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(emb_sum, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + (long)xi[j] * D;
            const float *pe = pos_emb  + (long)j * D;
            for (int m = 0; m < D; m++) emb_sum[m] += be[m] + pe[m];
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int m = 0; m < D; m++) sum += emb_sum[m] * W1[(long)m * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```