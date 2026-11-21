```hypothesis
Family: label-emb-mlp (positional embeddings + one ReLU hidden layer).
Why: Character prediction benefits from distributed representations; summing 16 positional embeddings captures n-gram statistics efficiently, and a hidden layer models interactions. Expected to beat label-linear (0.3375) with moderate inference cost (embedding lookups + two matrix multiplies).
Failure: If dev accuracy ≤ 0.3375 or instructions/item > 2× label-linear (~15k Ir/item).
```

```python
"""label-emb-mlp: positional embeddings + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, d, E, H, C; float32 embed[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 64          # embedding size
H = 128         # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 128
EPOCHS = 30
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    embed = (rng.normal(0, 0.02, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            # embeddings: (bs, d, E)
            emb = embed[:, xb.T, :].transpose(1, 0, 2)  # (bs, d, E)
            h_in = emb.sum(axis=1)                       # (bs, E)
            h_pre = h_in @ W1 + b1                       # (bs, H)
            h = np.maximum(h_pre, 0)                     # ReLU
            logits = h @ W2 + b2                         # (bs, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)   # (bs, C)

            # Backward
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h_pre > 0)
            grad_W1 = h_in.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_h_in = grad_h_pre @ W1.T                   # (bs, E)

            # Embedding gradients: scatter-add
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                idxs = xb[:, j]  # (bs,)
                np.add.at(grad_embed[j], idxs, grad_h_in)

            grads = [grad_embed, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * g * g
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Pack weights
    magic = 0x454D424C  # "LBME" little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    blob = header
    blob += embed.astype("<f4").tobytes()
    blob += W1.astype("<f4").tobytes()
    blob += b1.astype("<f4").tobytes()
    blob += W2.astype("<f4").tobytes()
    blob += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_in = malloc(sizeof(float) * E);
    float *h_pre = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        memset(h_in, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) h_in[e_idx] += e[e_idx];
        }

        // Hidden layer: h_pre = h_in @ W1 + b1, then ReLU
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e_idx = 0; e_idx < E; e_idx++) sum += h_in[e_idx] * W1[(long)e_idx * H + u];
            h_pre[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h_pre[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```