```hypothesis
We propose label-posemb: a position-wise embedding model that learns a distinct 32-dim embedding for each of the 16 context bytes, sums them, and applies a linear classifier. This captures positional n-gram information more efficiently than label-linear's one-hot linear layer (0.3375 acc) with only ~130k parameters, and should exceed label-linear's accuracy while keeping inference cost low (similar Ir/item). If dev accuracy is not above 0.34, the hypothesis fails.
```

```python
"""Position-wise embedding model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16 position-specific embedding tables (256x32), summed, then linear to 65 classes.
Trained with Adam (lr=3e-3, L2=1e-4), batch 64, 20 epochs.
Weights format: magic 0x54454D42, d=16, D=32, C=65; then embeddings[d][256][D], W[D][C], b[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split exploration)
D = 32
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x54454D42  # "TEMB"

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
    assert d == 16

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embeddings = [rng.normal(0, 1.0/math.sqrt(256), (256, D)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 1.0/math.sqrt(D), (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = embeddings + [W, b]
    # Adam states
    m_states = [np.zeros_like(p) for p in params]
    v_states = [np.zeros_like(p) for p in params]

    step = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: sum embeddings
            emb_sum = np.zeros((B, D), np.float32)
            for pos in range(d):
                emb_sum += embeddings[pos][xb[:, pos]]

            logits = emb_sum @ W + b            # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient on logits
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B                    # (B, C)

            # Gradients for W and b
            grad_W = emb_sum.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)

            # Gradient on emb_sum
            grad_emb_sum = grad_logits @ W.T    # (B, D)

            # Gradients for embeddings
            grad_embeddings = [np.zeros((256, D), np.float32) for _ in range(d)]
            for pos in range(d):
                np.add.at(grad_embeddings[pos], xb[:, pos], grad_emb_sum)
                grad_embeddings[pos] += L2 * embeddings[pos]

            grads = grad_embeddings + [grad_W, grad_b]

            # Adam update
            step += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m = m_states[i]
                v = v_states[i]
                m *= BETA1; m += (1 - BETA1) * g
                v *= BETA2; v += (1 - BETA2) * g * g
                m_hat = m / (1 - BETA1 ** step)
                v_hat = v / (1 - BETA2 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Pack weights
    header = struct.pack("<4i", MAGIC, d, D, C)
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in embeddings)
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Position-wise embedding inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x54454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || D <= 0 || D > 256 || C != 65) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(sum, 0, sizeof(float) * D);
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) sum[j] += e[j];
        }

        for (int c = 0; c < C; c++) {
            float acc = b[c];
            for (int j = 0; j < D; j++) acc += sum[j] * W[(long)j * C + c];
            logits[c] = acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```