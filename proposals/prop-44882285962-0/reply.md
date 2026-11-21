```hypothesis
Family: label-fm (factorization machine with shared embeddings for pairwise interactions).
Why: Adult has mixed ordinal/categorical features; FM captures second-order interactions efficiently with O(d*k) inference (k=8). Beats linear (0.8524) by modeling feature crosses, and is faster than MLP16 (2325 Ir/item) due to tiny k.
Failure: If dev accuracy <= 0.8524 or Ir/item >= 2000, the model doesn't justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses embeddings per feature value (256) of dim k=8.
Adam lr=0.01, L2=1e-4, batch=256, epochs=30. Exports lookup tables for embeddings and linear weights."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: for each feature j, 256 x K
    scale_emb = math.sqrt(6.0 / (256 + K))
    E = rng.uniform(-scale_emb, scale_emb, (d, 256, K)).astype(np.float32)
    # Linear weights: for each feature j, 256 x C
    scale_lin = math.sqrt(6.0 / (256 + C))
    W = rng.uniform(-scale_lin, scale_lin, (d, 256, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]  # (B, d)
            yb = y[idx]  # (B,)
            B = xb.shape[0]

            # Forward
            logits = np.zeros((B, C), np.float32)
            for j in range(d):
                logits += W[j, xb[:, j], :]

            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                emb = E[j, xb[:, j], :]  # (B, K)
                sum_emb += emb
                sum_emb_sq += emb * emb

            interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1, keepdims=True)  # (B, 1)
            logits += interaction
            logits += b

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backward
            grad_b = grad_logits.sum(axis=0)

            grad_W = np.zeros_like(W)
            for j in range(d):
                np.add.at(grad_W[j], xb[:, j], grad_logits)

            grad_E = np.zeros_like(E)
            for j in range(d):
                emb = E[j, xb[:, j], :]  # (B, K)
                interaction_grad = grad_logits.sum(axis=1)  # (B,)
                factor = interaction_grad[:, None] * (sum_emb - emb)  # (B, K)
                np.add.at(grad_E[j], xb[:, j], factor)

            grad_W += L2 * W
            grad_E += L2 * E
            grad_b += L2 * b

            t += 1
            # Update E
            mE = BETA1 * mE + (1 - BETA1) * grad_E
            vE = BETA2 * vE + (1 - BETA2) * grad_E * grad_E
            mE_hat = mE / (1 - BETA1 ** t)
            vE_hat = vE / (1 - BETA2 ** t)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + EPS)

            # Update W
            mW = BETA1 * mW + (1 - BETA1) * grad_W
            vW = BETA2 * vW + (1 - BETA2) * grad_W * grad_W
            mW_hat = mW / (1 - BETA1 ** t)
            vW_hat = vW / (1 - BETA2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + EPS)

            # Update b
            mb = BETA1 * mb + (1 - BETA1) * grad_b
            vb = BETA2 * vb + (1 - BETA2) * grad_b * grad_b
            mb_hat = mb / (1 - BETA1 ** t)
            vb_hat = vb / (1 - BETA2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + EPS)

    # Write weights.bin
    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += E.transpose(0,1,2).reshape(-1).astype("<f4").tobytes()
    out += W.transpose(0,1,2).reshape(-1).astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax).
 * Weights format: magic 0x464D4C31, int32 d, K, C; float32 E[d][256][K], W[d][256][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * K + (long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * K;
    const float *b = W + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int u = 0; u < K; u++) { sum_emb[u] = 0.0f; sum_emb_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *emb = E + ((long)j * 256 + v) * K;
            const float *lin = W + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) logits[c] += lin[c];
            for (int u = 0; u < K; u++) {
                float e = emb[u];
                sum_emb[u] += e;
                sum_emb_sq[u] += e * e;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < K; u++) {
            interaction += sum_emb[u] * sum_emb[u] - sum_emb_sq[u];
        }
        interaction *= 0.5f;
        for (int c = 0; c < C; c++) logits[c] += interaction;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```