```hypothesis
Family: label-fm (Factorization Machine). Adult income depends on feature interactions (e.g., education × occupation); FM captures pairwise interactions via low-rank embeddings (k=8) with O(d·k) inference, beating linear (0.8524) while staying far faster than GBRT/MLP. Failure if dev accuracy ≤ 0.8524 or Ir/item > 1000.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. FM with shared embeddings, class-specific linear terms.
Embedding dim k=8. Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: int32 magic 0x464D4C31, d, k, C; float32 w0[C], linear[C][d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

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

    w0 = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    embed = (rng.normal(0, 1.0/math.sqrt(K), (d, 256, K))).astype(np.float32)

    params = [w0, linear, embed]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            scores = w0[None, :].copy()
            for j in range(d):
                scores += linear[:, j, xb[:, j]].T

            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                e = embed[j, xb[:, j]]
                sum_emb += e
                sum_emb_sq += e * e
            interaction = 0.5 * (np.sum(sum_emb * sum_emb, axis=1) - np.sum(sum_emb_sq, axis=1))
            scores += interaction[:, None]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = (probs / B).astype(np.float32)

            grad_w0 = grad_scores.sum(axis=0)
            grad_linear = np.zeros_like(linear)
            grad_embed = np.zeros_like(embed)

            for c in range(C):
                for j in range(d):
                    np.add.at(grad_linear[c, j], xb[:, j], grad_scores[:, c])

            d_interaction = grad_scores.sum(axis=1)
            for j in range(d):
                e = embed[j, xb[:, j]]
                g = d_interaction[:, None] * (sum_emb - e)
                vals = xb[:, j]
                for u in range(K):
                    np.add.at(grad_embed[j, :, u], vals, g[:, u])

            grad_w0 += L2 * w0
            grad_linear += L2 * linear
            grad_embed += L2 * embed

            grads = [grad_w0, grad_linear, grad_embed]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    MAGIC = 0x464D4C31
    header = struct.pack("<4i", MAGIC, d, K, C)
    out = header
    out += w0.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class index (argmax).
 * Weights: int32 magic 0x464D4C31, d, k, C; float32 w0[C], linear[C][d][256], embed[d][256][k]. */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32 || d < 1 || d > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * (2 + 2L * d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 4);
    float *w0 = ptr; ptr += 2;
    float *linear = ptr; ptr += 2 * d * 256;
    float *embed = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int u = 0; u < k; u++) { sum_emb[u] = 0.0f; sum_emb_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++)
                scores[c] += linear[(c * d + j) * 256 + val];
            const float *e = embed + ((j * 256 + val) * k);
            for (int u = 0; u < k; u++) {
                float v = e[u];
                sum_emb[u] += v;
                sum_emb_sq[u] += v * v;
            }
        }

        float interaction = 0.0f;
        for (int u = 0; u < k; u++)
            interaction += sum_emb[u] * sum_emb[u] - sum_emb_sq[u];
        interaction *= 0.5f;

        for (int c = 0; c < C; c++) scores[c] += interaction;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```