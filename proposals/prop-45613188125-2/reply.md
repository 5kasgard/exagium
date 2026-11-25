```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (E=8), concatenates them, and passes through a single ReLU hidden layer (H=64) to predict next character. Captures pairwise interactions missed by linear one-hot model while keeping parameter count low (~45k). If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 8000 Ir/item, the hypothesis fails.
```

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen based on tuning split intuition)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    d = card["input"]["bytes"]  # 16
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y_bytes = b"".join(records(sys.argv[3]))
    y = np.frombuffer(y_bytes, dtype=np.uint8)  # each record is 1 byte class index
    C = card["classes"]  # 65
    n = X.shape[0]
    assert y.shape[0] == n

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: for each position j, a matrix 256 x E
    embeddings = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # Hidden layer weights: (d*E) x H
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer weights: H x C
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]  # (B, d)
            y_batch = y[idx]  # (B,)
            B = len(idx)

            # Forward: embeddings lookup
            emb_out = np.zeros((B, d * E), dtype=np.float32)
            for j in range(d):
                emb_out[:, j*E:(j+1)*E] = embeddings[j, x_batch[:, j], :]

            # Hidden layer
            hidden = emb_out @ W1 + b1  # (B, H)
            hidden_relu = np.maximum(hidden, 0)

            # Output layer
            logits = hidden_relu @ W2 + b2  # (B, C)

            # Cross-entropy loss and gradients
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            grad_logits = probs
            grad_logits[np.arange(B), y_batch] -= 1.0
            grad_logits /= B

            # Backprop to output layer
            grad_W2 = hidden_relu.T @ grad_logits + L2 * W2
            grad_b2 = np.sum(grad_logits, axis=0)

            # Backprop to hidden layer
            grad_hidden_relu = grad_logits @ W2.T
            grad_hidden = grad_hidden_relu * (hidden > 0)

            # Backprop to W1, b1
            grad_W1 = emb_out.T @ grad_hidden + L2 * W1
            grad_b1 = np.sum(grad_hidden, axis=0)

            # Backprop to embeddings
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                grad_emb_j = np.zeros((256, E), dtype=np.float32)
                np.add.at(grad_emb_j, x_batch[:, j], grad_hidden[:, j*E:(j+1)*E])
                grad_embeddings[j] = grad_emb_j + L2 * embeddings[j]

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Format: magic (int32), d, E, H, C (int32), then embeddings (d*256*E float32), W1 ((d*E)*H float32), b1 (H float32), W2 (H*C float32), b2 (C float32)
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += embeddings.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20 + 4L * (d * 256 * E + d * E * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embeddings = (const float *)(wbuf + 20);
    const float *W1 = embeddings + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float emb_concat[d * E];
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + (j * 256 + xi[j]) * E;
            memcpy(emb_concat + j * E, emb, E * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) {
                sum += emb_concat[k] * W1[k * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(hidden);
    free(scores);
    if (x) free(x);
    free(wbuf);
    return 0;
}
```