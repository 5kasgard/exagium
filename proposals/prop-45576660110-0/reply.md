```hypothesis
We propose "label-emb", an embedding-based logistic regression that learns a dense vector (size 4) for each of the 256 possible values per feature, then classifies via a linear layer. This captures feature interactions implicitly through embeddings and should outperform label-linear on this tabular dataset with many categorical features, while keeping inference cost low (only embedding lookups and a small matrix-vector product). If dev accuracy does not exceed label-linear's 0.8524 by at least 0.005, or if inference cost exceeds label-mlp16's ~2300 Ir/item, the approach fails.
```

```python
"""label-emb: embedding logistic regression for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy+stdlib only.
Exports: magic 0x454D424C, d, E, C; float32 embeddings[d][256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 4                 # embedding dimension per feature
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 20


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

    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # linear layer: (d*E, C)
    W = (rng.normal(0, 0.01, (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward: gather embeddings and concat
            # xb: (bs, d) uint8
            # we want vec: (bs, d*E)
            vec = np.zeros((bs, d * E), np.float32)
            for j in range(d):
                vec[:, j*E:(j+1)*E] = emb[j, xb[:, j], :]

            logits = vec @ W + b                        # (bs, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs                    # (bs, C)

            # gradients
            grad_vec = grad_logits @ W.T                # (bs, d*E)
            grad_W = vec.T @ grad_logits + L2 * W       # (d*E, C)
            grad_b = grad_logits.sum(axis=0)            # (C,)

            # scatter grad_vec back to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # grad_vec[:, j*E:(j+1)*E] has shape (bs, E)
                # xb[:, j] has indices
                np.add.at(grad_emb[j], xb[:, j], grad_vec[:, j*E:(j+1)*E])

            # Adam updates
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W, grad_W, m_W, v_W),
                (b, grad_b, m_b, v_b)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights: magic, d, E, C; then embeddings, W, b (all float32 little-endian)
    magic = 0x454D424C  # "LMBE" in little-endian
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index.
 * Weights format: int32 magic(0x454D424C), d, E, C; float32 embeddings[d][256][E], W[d*E][C], b[C]. */
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
    if (C < 1 || C > 256 || d < 1 || E < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(vec + j * E, e, E * sizeof(float));
        }

        // linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * E; k++) {
            float v = vec[k];
            if (v != 0.0f) {
                const float *w = W + (long)k * C;
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