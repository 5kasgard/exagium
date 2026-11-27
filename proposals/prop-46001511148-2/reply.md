```hypothesis
Family: label-embedding-mlp. Position-specific byte embeddings (dim 16) are summed across the 16 context positions, then passed through a ReLU hidden layer (64 units) and a linear output layer (65 classes). This captures pairwise position interactions via the hidden layer while keeping inference fast due to the small embedding dimension. It should beat label-linear (0.3375) because label-linear cannot model interactions between positions. If dev accuracy does not exceed 0.34, the hypothesis fails.
```
```python
"""label-embedding-mlp: position-specific byte embeddings summed, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on tuning split)
E = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4244  # "EMBD" in little-endian


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
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = rng.normal(0.0, 1.0 / math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    # W1: (E, H), b1: (H,)
    W1 = rng.uniform(-1, 1, (E, H)).astype(np.float32) * math.sqrt(6.0 / (E + H))
    b1 = np.zeros(H, dtype=np.float32)
    # W2: (H, C), b2: (C,)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / (H + C))
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)

            # Forward: sum embeddings
            # emb_sum shape (batch, E)
            emb_sum = np.zeros((len(yb), E), dtype=np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j], :]

            # Hidden layer
            hidden_pre = emb_sum @ W1 + b1
            hidden = np.maximum(hidden_pre, 0.0)

            # Output logits
            logits = hidden @ W2 + b2

            # Softmax cross-entropy gradient
            # stable softmax
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)   # (batch, C)

            # Backprop
            # grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0   # ReLU gradient

            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)

            # Gradient for embeddings: for each position j, scatter-add grad_hidden to emb[j, xb[:,j], :]
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # grad_hidden shape (batch, H) but we need (batch, E) for emb gradient? Wait:
                # emb_sum = sum_j emb[j, xb[:,j], :]
                # So d(emb_sum)/d(emb[j, v, :]) = 1 if xb[i,j]==v else 0.
                # grad_emb[j, v, :] = sum_{i: xb[i,j]==v} grad_hidden[i] @ W1.T? Actually:
                # hidden = emb_sum @ W1 + b1, so d hidden / d emb_sum = W1^T.
                # grad_emb_sum = grad_hidden @ W1.T  (batch, E)
                # Then grad_emb[j, xb[i,j], :] += grad_emb_sum[i]
                pass  # We'll compute grad_emb_sum first.

            grad_emb_sum = grad_hidden @ W1.T   # (batch, E)
            for j in range(d):
                # Use np.add.at for scatter add
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic (int32), d, E, H, C (int32 each)
    # Then embeddings (d*256*E float32), W1 (E*H), b1 (H), W2 (H*C), b2 (C) all float32 little-endian.
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # Flatten in order
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
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero emb_sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // Sum embeddings for each position
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // Hidden layer: ReLU(emb_sum * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output logits: hidden * W2 + b2
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

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```