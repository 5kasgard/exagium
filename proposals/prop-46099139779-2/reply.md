```hypothesis
Family: label-embed-linear. Learns an 8-d embedding per byte position (16 positions) followed by a linear layer to 65 classes. This reduces parameters (41k vs 266k for label-linear) and captures byte similarities, improving generalization on this small dataset. Expect dev accuracy > 0.34 (label-linear is 0.3375) with lower inference cost (smaller matrix multiply). If dev accuracy ≤ 0.34, the hypothesis fails.
```

```python
"""label-embed-linear: embedding per position + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (no tuning split used)
E = 8                 # embedding dimension per position
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000      # max samples seen (like label-linear)

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
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: for each position, 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    # Linear layer: (d*E) x C
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E + C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    # Precompute one-hot indices for embedding lookup? We'll do batched gather.
    # For speed, we can precompute embedding matrix flattened? But d=16 small.
    # We'll just use fancy indexing in the loop.

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: embeddings -> concat -> linear
            # emb[xb[:,j], j] but we need (B, E) per position
            # We'll do a loop over positions (d=16) for clarity and low overhead.
            emb_concat = np.empty((B, d * E), dtype=np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]

            logits = emb_concat @ W + b                        # (B, C)
            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)       # (B, C)

            # Gradients
            grad_W = emb_concat.T @ grad_logits + L2 * W       # (d*E, C)
            grad_b = grad_logits.sum(axis=0)                   # (C,)
            grad_emb_concat = grad_logits @ W.T                # (B, d*E)

            # Backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                g = grad_emb_concat[:, j*E:(j+1)*E]            # (B, E)
                # Scatter-add: for each sample, add g[i] to grad_emb[j, xb[i, j]]
                np.add.at(grad_emb[j], xb[:, j], g)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # Write weights.bin
    # Format: magic (0x454D424C), d, E, C, then embeddings [d][256][E], W [d*E][C], b [C] (all float32 little-endian)
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-linear inference: ./infer weights.bin in.rec N
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 16 + (long)d * 256 * E * 4 + (long)d * E * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wb + 16);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Linear layer: scores = concat @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int fe = 0; fe < d * E; fe++) {
            float v = concat[fe];
            if (v == 0.0f) continue;
            const float *wcol = W + (long)fe * C;
            for (int c = 0; c < C; c++) scores[c] += v * wcol[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```