```hypothesis
Family: label-embedlinear (shared embedding + linear). Why: Character-level language modeling benefits from shared representations of characters across positions; a 16-byte context with 65 classes has 266k parameters in label-linear. A shared embedding table (256×E) followed by a linear layer (16E×65) reduces parameters to ~17k (E=16), improving generalization on 24k training examples. The linear layer can still learn position-specific combinations. Failure mode: if positional information is essential and cannot be linearly separated from the shared embeddings, accuracy will not exceed label-linear.
```

```python
"""label-embedlinear: shared embedding + linear for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, d, E, C; float32 emb[256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embedding table: 256 x E
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # Linear layer: (d*E) x C
    k = d * E
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: gather embeddings
            # xb shape (bs, d) -> for each position j, emb[xb[:,j]] shape (bs, E)
            # concatenate to (bs, d*E)
            emb_seq = np.zeros((bs, k), np.float32)
            for j in range(d):
                emb_seq[:, j*E:(j+1)*E] = emb[xb[:, j]]

            logits = emb_seq @ W + b  # (bs, C)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)  # (bs, C)

            # Gradients
            grad_W = emb_seq.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            # Gradient wrt emb_seq
            grad_emb_seq = grad_logits @ W.T  # (bs, k)
            # Scatter to embedding table
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_emb_seq[:, j*E:(j+1)*E])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x454D424C  # 'LMBE'
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedlinear infer: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights: magic, d, E, C; float32 emb[256][E], W[d*E][C], b[C]. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long k = (long)d * E;
    long expected = 16 + 4L * (256L * E + k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + 256L * E;
    const float *b = W + k * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *seq = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(seq + j * E, e, E * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < k; idx++) {
            float v = seq[idx];
            if (v != 0.0f) {
                const float *w = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```