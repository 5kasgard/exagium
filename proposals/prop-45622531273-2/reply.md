```hypothesis
The label-embmlp family learns a small embedding per position (16×256×E) followed by a one-hidden-layer MLP (E·16 → H → 65). This captures pairwise interactions between context bytes that a linear model misses, while staying far smaller than a full one-hot MLP. If dev accuracy does not exceed label-linear's 0.3375, the extra parameters and non-linearity are not beneficial for this dataset size.
```

```python
"""Embedding + one-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4C50, d, E, H, C; then float32 arrays:
  emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
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
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: concatenate embeddings
            # emb shape (d,256,E) -> for each sample, gather rows xb[:,j] -> (bs, d, E) -> (bs, d*E)
            emb_concat = emb[np.arange(d), xb.T].transpose(1, 0, 2).reshape(bs, d * E)  # (bs, d*E)
            h_pre = emb_concat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h_pre > 0)
            grad_W1 = emb_concat.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)

            # Gradient for embeddings: grad_h_pre @ W1^T -> (bs, d*E) -> reshape to (bs, d, E)
            grad_emb_concat = grad_h_pre @ W1.T  # (bs, d*E)
            grad_emb_concat = grad_emb_concat.reshape(bs, d, E)  # (bs, d, E)
            # Accumulate per (position, byte)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                bytes_j = xb[:, j]
                np.add.at(grad_emb[j], bytes_j, grad_emb_concat[:, j, :])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # header: magic, d, E, H, C
    header = struct.pack("<5i", 0x454D4C50, d, E, H, C)
    out = bytearray(header)
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
/* Embedding + one-hidden-layer MLP inference: ./infer weights.bin in.rec N
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4C50) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 20
        + (size_t)d * 256 * E * 4
        + (size_t)d * E * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + (size_t)d * 256 * E;
    const float *b1 = W1 + (size_t)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc((size_t)d * E * sizeof(float));
    float *hidden = malloc((size_t)H * sizeof(float));
    float *logits = malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(size_t)k * H + h];
            hidden[h] = sum > 0 ? sum : 0;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    free(wb); if (x) free(x);
    return 0;
}
```