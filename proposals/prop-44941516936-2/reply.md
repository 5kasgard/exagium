```hypothesis
Family: label-embed-linear (factorized linear with per-position byte embeddings).
Why: Character-level language modeling benefits from shared representations of similar bytes; embeddings reduce parameters (82k vs 266k) and act as regularization, improving generalization over one-hot linear. The factorized structure also cuts inference FLOPs (16*E*C vs d*256*C) and memory.
Failure: If dev accuracy <= 0.3375 (label-linear) or latency exceeds label-linear by >20%.
```

```python
"""label-embed-linear: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, int32 d, E, C; float32 embeddings[d][256][E], W[C][d*E], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
E = 8  # embedding dimension


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

    # parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) / math.sqrt(E)).astype(np.float32)
    W = (rng.uniform(-1, 1, (C, d * E)) * math.sqrt(6.0 / (d * E + C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: concatenate embeddings
            z = np.empty((bsz, d * E), np.float32)
            for j in range(d):
                z[:, j * E:(j + 1) * E] = emb[j, xb[:, j]]

            logits = z @ W.T + b  # (bsz, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient on logits
            g = probs
            g[np.arange(bsz), yb] -= 1.0
            g /= bsz

            # grad_W, grad_b
            grad_W = g.T @ z + L2 * W
            grad_b = g.sum(axis=0)

            # grad_z = g @ W  (bsz, d*E)
            grad_z = g @ W

            # grad_emb: scatter-add per position
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                gj = grad_z[:, j * E:(j + 1) * E]
                np.add.at(grad_emb[j], xb[:, j], gj)
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

    # pack weights
    magic = 0x454D424C
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index.
 * Weights: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W[C][d*E], b[C]. */
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
    if (C < 1 || C > 256 || d < 1 || E < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * d * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)C * d * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    if (!scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int c = 0; c < C; c++) {
                const float *w = W + (long)c * d * E + j * E;
                float sum = 0.0f;
                for (int k = 0; k < E; k++) sum += e[k] * w[k];
                scores[c] += sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```