```hypothesis
Family: label-embedding. Learns a dense embedding (dim D=32) for each of the 16 context positions × 256 byte values, sums them, and applies a linear classifier to 65 classes. Captures character similarities (e.g., letters vs punctuation) that one-hot linear models miss, while keeping inference a single matrix-vector multiply. Expected to beat label-linear (0.3375) by ≥1% absolute accuracy with similar CPU cost. Failure: dev accuracy ≤0.3475.
```

```python
"""label-embedding: position-specific byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D4244, int32 d,D,C; float32 emb[d][256][D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 32                  # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 10


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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: (d, 256, D)
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W = (rng.normal(0, 0.01, (D, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam parameters
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # forward: sum embeddings
            # emb_sum[b, k] = sum_j emb[j, xb[b,j], k]
            emb_sum = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j], :]   # (B, D)

            logits = emb_sum @ W + b             # (B, C)
            # stable softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient on logits
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # gradients
            grad_W = emb_sum.T @ grad_logits + L2 * W          # (D, C)
            grad_b = grad_logits.sum(axis=0)                   # (C,)
            grad_emb_sum = grad_logits @ W.T                   # (B, D)

            # backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # accumulate grad_emb[j, byte, :] += grad_emb_sum[b, :] for each b where xb[b,j]==byte
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            t += 1
            # Adam updates
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W, grad_W, m_W, v_W),
                (b, grad_b, m_b, v_b),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    magic = 0x454D4244  # 'EMBD' little-endian
    header = struct.pack("<4i", magic, d, D, C)
    body = emb.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x454D4244, int32 d,D,C; float32 emb[d][256][D], W[D][C], b[C]. */
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
    if (d != 16 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(emb_sum, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int k = 0; k < D; k++) emb_sum[k] += e[k];
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < D; k++) sum += emb_sum[k] * W[(long)k * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```