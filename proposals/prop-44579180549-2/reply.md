```hypothesis
Family: label-embed (per-position byte embeddings + linear classifier).
Why: The 16-byte context consists of categorical bytes; one-hot linear (label-linear) uses 4096 features but ignores similarities between characters. Embedding each byte into a low-dimensional space (D=8) captures semantic similarity (e.g., letters near each other) and reduces parameters to ~41k vs 266k, which should improve generalization and speed. The model is still a simple two-layer network trainable with Adam.
Failure signal: If dev accuracy does not exceed label-linear's 0.3375, it indicates that the task requires explicit high-dimensional position-specific interactions that embeddings cannot capture.
```

```python
"""label-embed: per-position byte embeddings followed by a linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, int32 d, D, C; float32 E[d][256][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D4244  # 'EMBD' in little-endian byte order


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
    # Embeddings: (d, 256, D)
    E = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    # Output layer: (d*D, C)
    W = rng.normal(0, 0.01, (d * D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            # emb shape (B, d, D)
            emb = E[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)
            flat = emb.reshape(B, -1)                 # (B, d*D)
            logits = flat @ W + b                     # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B                          # (B, C)

            # Gradients for W and b
            grad_W = flat.T @ grad_logits + L2 * W    # (d*D, C)
            grad_b = grad_logits.sum(axis=0)          # (C,)

            # Gradient for embeddings
            grad_flat = grad_logits @ W.T             # (B, d*D)
            grad_emb = grad_flat.reshape(B, d, D)     # (B, d, D)
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_emb[:, j])
            grad_E += L2 * E

            # Adam update
            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * gr * gr
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output is one byte: predicted class index.
 * Weights format: magic 0x454D4244, int32 d, D, C; float32 E[d][256][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || D != 8 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float flat[128];               // d*D = 16*8 = 128
        for (int j = 0; j < d; j++) {
            const float *emb = E + ((long)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) flat[j * D + kk] = emb[kk];
        }

        float scores[65];              // C = 65
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < d * D; f++) {
            float val = flat[f];
            if (val != 0.0f) {
                const float *wrow = W + (long)f * C;
                for (int c = 0; c < C; c++) scores[c] += val * wrow[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```