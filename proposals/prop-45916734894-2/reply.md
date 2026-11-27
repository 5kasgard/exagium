```hypothesis
Family: label-embedding. Factorizes the per-position one-hot linear weights into 16
position-specific embedding tables (256 × E) and a shared linear head (16·E → 65).
This reduces parameters from 266k to ~164k (E=32), acts as strong regularization
for the small dataset, and should improve accuracy over label-linear. If dev accuracy
≤ 0.3375, the factorization is too restrictive.
```

```python
"""label-embedding: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports weights in the label-linear format (magic 0x54424C31, H=0) so the same
infer.c works."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, last 20% of train.in)
E = 32               # embedding dimension
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 256
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
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: list of d arrays, each (256, E)
    embeddings = []
    for _ in range(d):
        # Xavier uniform
        scale = math.sqrt(6.0 / (256 + E))
        emb = rng.uniform(-scale, scale, size=(256, E)).astype(np.float32)
        embeddings.append(emb)

    # Initialize linear head: W (d*E, C), b (C,)
    scale = math.sqrt(6.0 / (d * E + C))
    W = rng.uniform(-scale, scale, size=(d * E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    params = embeddings + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Pre-split W into per-position matrices for convenience
    # W_j = W[j*E : (j+1)*E, :]
    Ws = [W[j*E:(j+1)*E] for j in range(d)]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yy = y[idx]                # (B,)
            B = len(yy)

            # Forward: compute logits
            logits = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                # embeddings[xb[:, j]] -> (B, E)
                emb_batch = embeddings[j][xb[:, j]]
                logits += emb_batch @ Ws[j]
            logits += b

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Gradients
            grad_embeddings = [np.zeros_like(emb) for emb in embeddings]
            grad_Ws = [np.zeros_like(Wj) for Wj in Ws]
            grad_b = dlogits.sum(axis=0)

            for j in range(d):
                # Gradient for W_j: emb_batch.T @ dlogits
                emb_batch = embeddings[j][xb[:, j]]    # (B, E)
                grad_Ws[j] = emb_batch.T @ dlogits
                # Gradient for embeddings: for each sample, add dlogits[i] @ W_j.T to the used embedding row
                grad_emb_batch = dlogits @ Ws[j].T     # (B, E)
                # Accumulate using np.add.at
                np.add.at(grad_embeddings[j], xb[:, j], grad_emb_batch)

            # L2 regularization
            for j in range(d):
                grad_Ws[j] += L2 * Ws[j]
                grad_embeddings[j] += L2 * embeddings[j]

            # Flatten gradients in the same order as params
            grads = grad_embeddings + [np.vstack(grad_Ws), grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, compute the effective per-position per-byte linear weights T[j][v] (256, C)
    T = []
    for j in range(d):
        T.append(embeddings[j] @ Ws[j])   # (256, C)
    T = np.stack(T, axis=0).astype(np.float32)   # (d, 256, C)
    b1 = b.astype(np.float32)

    # Write weights in label-linear format (H=0)
    magic = 0x54424C31  # 'TBL1'
    out = struct.pack("<4i", magic, d, 0, C)
    out += T.tobytes()
    out += b1.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Format: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H: float32 W2[H][C], b2[C]
 * (W = H ? H : C). This infer.c is identical to the label-linear one. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```