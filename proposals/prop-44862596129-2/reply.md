```hypothesis
We propose "label-factorized", a factorized linear model where each (position, byte) pair maps to a low-dimensional embedding (E=32), summed across the 16 positions, then linearly projected to 65 classes. This captures position-specific byte statistics with fewer parameters than label-linear (133k vs 266k), potentially generalizing better on the 24k training examples. Inference cost is only 16*32 + 32*65 multiply-adds per item, similar to label-linear. If dev accuracy does not exceed 0.34, the factorization is too restrictive for this task.
```

```python
"""Factorized linear model for next-byte prediction: each (position, byte) has an E-dim embedding,
summed over positions, then linear to classes. Trained with Adam, deterministic seed 1."""
import json, math, struct, sys

import numpy as np

E = 32                  # embedding dimension (tuned on last 20% of train.in)
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-5, 64, 15

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
    rng = np.random.default_rng(SEED)

    # parameters: emb[d, 256, E], W_out[E, C], b_out[C]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W_out = (rng.normal(0, 0.01, (E, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # forward: sum embeddings
            summed = np.zeros((bs, E), np.float32)
            for j in range(d):
                summed += emb[j, xb[:, j], :]
            logits = summed @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # loss gradient
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs

            # gradients
            grad_W = summed.T @ grad_logits + L2 * W_out
            grad_b = grad_logits.sum(axis=0)
            grad_summed = grad_logits @ W_out.T

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_summed)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W_out, grad_W, m_W, v_W),
                (b_out, grad_b, m_b, v_b)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights: magic, d, E, C; then emb (d*256*E), W_out (E*C), b_out (C)
    magic = 0x46414354  # "FACT"
    header = struct.pack("<4i", magic, d, E, C)
    body = emb.astype("<f4").tobytes() + W_out.astype("<f4").tobytes() + b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(summed, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < E; u++) summed[u] += e[u];
        }
        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int u = 0; u < E; u++) {
            float su = summed[u];
            if (su == 0.0f) continue;
            const float *w = W_out + (long)u * C;
            for (int c = 0; c < C; c++) logits[c] += su * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(summed); free(logits);
    return 0;
}
```