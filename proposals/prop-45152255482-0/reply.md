```hypothesis
Family: label-emblin (embedding + linear). All 14 features are already quantized to 256 bins; we learn an 8-dim embedding per feature, concatenate them (112-dim), and apply a linear classifier. This captures non-linear feature interactions via the embedding space while keeping inference a single matrix-vector multiply. It should beat label-linear (0.8524) on dev because embeddings model feature similarities that one-hot cannot. If dev accuracy ≤ 0.8524, the hypothesis fails.
```
```python
"""label-emblin: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format (little-endian):
  int32 magic=0x454D424C, d, E, C
  float32 emb[d][256][E]   (row-major per feature)
  float32 W[d*E][C]        (row-major)
  float32 b[C]"""
import json, math, struct, sys

import numpy as np

E = 8
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    # linear weights: (d*E, C)
    W = (rng.normal(0, 1.0 / math.sqrt(d * E), (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def sg(scores, yy):
        scores -= scores.max(axis=1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = xb.shape[0]

            # gather embeddings: (B, d, E) -> (B, d*E)
            xb_emb = emb[:, xb]              # (d, B, E)
            xb_emb = xb_emb.transpose(1, 0, 2).reshape(B, d * E)  # (B, d*E)

            scores = xb_emb @ W + b
            g = sg(scores, yb)               # (B, C)

            # gradients
            grad_W = xb_emb.T @ g + L2 * W   # (d*E, C)
            grad_b = g.sum(0)                # (C,)
            grad_emb = np.zeros_like(emb)    # (d, 256, E)
            # backprop to embeddings: (B, d*E) @ W.T -> (B, d*E) then scatter
            g_emb_flat = g @ W.T             # (B, d*E)
            g_emb = g_emb_flat.reshape(B, d, E)  # (B, d, E)
            # scatter-add: for each feature j, add g_emb[:, j, :] to grad_emb[j, xb[:, j], :]
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], g_emb[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # write weights
    out = struct.pack("<4i", 0x454D424C, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emblin infer: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weight format: int32 magic=0x454D424C, d, E, C; float32 emb[d][256][E], W[d*E][C], b[C]. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long emb_sz = (long)d * 256 * E;
    long W_sz = (long)d * E * C;
    long expected = 16 + 4L * (emb_sz + W_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }

        // linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * E; idx++) {
            float v = concat[idx];
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += v * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```