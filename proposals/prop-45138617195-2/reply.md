```hypothesis
Family: label-emb (embedding + linear). Each of the 16 context positions gets a learned 8-d embedding; the 128-d concatenated vector feeds a linear layer to 65 classes. Embeddings capture character similarities (e.g., letters vs punctuation) and drastically reduce parameters vs. one-hot linear (41k vs 266k), so it should generalize better at lower compute. If dev accuracy ≤ 0.3375 (label-linear) or instructions/item exceed label-linear by >20%, the hypothesis fails.
```
```python
"""label-emb: embedding + linear for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension per position
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: list of d arrays, each (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # Linear weights: (d*E, C) and bias (C,)
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = emb + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: gather embeddings and concatenate
            x_emb = np.empty((B, d * E), np.float32)
            for j in range(d):
                x_emb[:, j * E:(j + 1) * E] = emb[j][xb[:, j]]

            logits = x_emb @ W + b                 # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient of cross-entropy
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Gradients for W and b
            grad_W = x_emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)

            # Gradients for embeddings
            grad_x_emb = grad_logits @ W.T          # (B, d*E)
            grad_emb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                slice_ = grad_x_emb[:, j * E:(j + 1) * E]
                # Accumulate gradients per embedding index
                np.add.at(grad_emb[j], xb[:, j], slice_)

            grads = grad_emb + [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, C, then embeddings (d * 256 * E floats), then W (d*E*C), then b (C)
    magic = 0x454D424C  # "LBME" in little-endian
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in emb)
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```
```c
/* label-emb: embedding + linear inference.
 * ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || E != 8 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * (d * 256 * E + d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);                     // [d][256][E]
    const float *W = emb + d * 256 * E;                             // [d*E][C]
    const float *b = W + d * E * C;                                 // [C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k_idx = 0; k_idx < d * E; k_idx++) {
            float xv = x_emb[k_idx];
            if (xv == 0.0f) continue;
            const float *w = W + k_idx * C;
            for (int c = 0; c < C; c++) scores[c] += xv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```