```hypothesis
We propose a "label-embedding" family: learn a low-dimensional embedding for each byte value at each of the 16 context positions, concatenate them, and apply a linear classifier. This captures positional character patterns with far fewer parameters than one-hot linear models, improving generalization on the small tinyshakespeare dataset. If dev accuracy does not exceed label-linear (0.3375), the low-rank assumption is invalid for this task.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
D = 16          # embedding dimension
LR = 0.01       # learning rate
L2 = 1e-4       # weight decay
BATCH = 256     # minibatch size
EPOCHS = 10     # training epochs
SEED = 1        # fixed seed for determinism

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
    n, d = X.shape
    C = card["classes"]
    assert d == 16
    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, D]
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # Linear weights: [C, d*D]
    W = rng.normal(0, 0.01, size=(C, d * D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    indices = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(indices)
        for start in range(0, n, BATCH):
            idx = indices[start:start + BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Concatenate embeddings
            concat = np.zeros((B, d * D), dtype=np.float32)
            for j in range(d):
                concat[:, j * D:(j + 1) * D] = emb[j, x_batch[:, j], :]

            # Forward pass
            logits = concat @ W.T + b                     # (B, C)
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradient of cross-entropy
            grad_logits = probs
            grad_logits[np.arange(B), y_batch] -= 1.0
            grad_logits /= B

            # Gradients for W and b
            grad_W = grad_logits.T @ concat + L2 * W      # (C, d*D)
            grad_b = np.sum(grad_logits, axis=0)

            # Gradient for embeddings
            grad_concat = grad_logits @ W                 # (B, d*D)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], x_batch[:, j], grad_concat[:, j * D:(j + 1) * D])
            grad_emb += L2 * emb

            # Adam updates
            t += 1
            # Embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
            W -= LR * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
            b -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    # Write weights.bin: magic, d, D, C; then embeddings, W, b (all float32 little-endian)
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)C * d * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * d * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* concatenate embeddings */
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            float *dst = concat + j * D;
            for (int kk = 0; kk < D; kk++) dst[kk] = src[kk];
        }

        /* linear layer: scores = W * concat + b */
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *wrow = W + (long)c * d * D;
            for (int kk = 0; kk < d * D; kk++) sum += wrow[kk] * concat[kk];
            scores[c] = sum;
        }

        /* argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```