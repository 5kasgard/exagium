```hypothesis
Family: label-embed (factorized linear model with per-position byte embeddings).
Why: The input is 16 categorical bytes; a linear model on one-hot features (label-linear) has 266k parameters and may overfit. Replacing one-hot with learned embeddings (8 dims each) reduces parameters to ~41k, adds a low-rank inductive bias that captures similarity between byte values, and retains positional information via concatenation. This should improve generalization on the small tinyshakespeare training set (24k samples) while keeping inference fast (~10k Ir/item).
Failure: If dev accuracy <= 0.3375 (label-linear) or CPU cost > 20k Ir/item, the factorization is not beneficial for this task.
```

```python
"""Factorized linear model with per-position byte embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

EMBEDDING_DIM = 8
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    E = EMBEDDING_DIM

    # Initialization
    emb = (rng.normal(0, 1.0 / 16.0, (d, 256, E))).astype(np.float32)
    W   = (rng.normal(0, 1.0 / math.sqrt(E * d), (E * d, C))).astype(np.float32)
    b   = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W,   v_W   = np.zeros_like(W),   np.zeros_like(W)
    m_b,   v_b   = np.zeros_like(b),   np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward: concatenate embeddings
            concat = np.zeros((B, E * d), np.float32)
            for j in range(d):
                concat[:, j * E:(j + 1) * E] = emb[j, xb[:, j], :]

            logits = concat @ W + b                        # (B, C)
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient of cross-entropy
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W = concat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_concat = grad_logits @ W.T                # (B, E*d)

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_concat[:, j * E:(j + 1) * E])
            grad_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in ((emb, grad_emb, m_emb, v_emb),
                                      (W,   grad_W,   m_W,   v_W),
                                      (b,   grad_b,   m_b,   v_b)):
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights: magic, d, E, C, emb[d][256][E], W[E*d][C], b[C] (all float32 LE)
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model with per-position byte embeddings.
 * ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: one byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *rec = buf + *pos + 4; *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(hdr, 1, 4, f); fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * d * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W   = emb + (long)d * 256 * E;
    const float *b   = W   + (long)E * d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * E * d);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < E * d; k++)
                sum += concat[k] * W[(long)k * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```