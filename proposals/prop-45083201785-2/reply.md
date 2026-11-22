```hypothesis
Family: label-emb (positional embeddings + linear). The 16-byte context benefits from dense embeddings that capture byte similarities (e.g., letters cluster in ASCII) while retaining position via separate embedding tables per position. This reduces parameters vs. one-hot linear (16*256*E + 16*E*65 vs 16*256*65) and should generalize better. If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear, the approach fails.
```

```python
"""label-emb: positional embeddings + linear for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D424C, d=16, E, C=65; float32 embeddings[d][256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 32                  # embedding dimension
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
    assert d == 16 and C == 65

    # Use last 20% as tuning split (by file order) for any hyper-parameter choices; here we fix E=32.
    n = X.shape[0]
    tune_start = int(0.8 * n)
    # We train on all data as per contract.

    rng = np.random.default_rng(SEED)
    # Embeddings: [d, 256, E] initialized with Xavier normal
    emb = (rng.normal(0, math.sqrt(2.0 / (256 + E)), size=(d, 256, E))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(2.0 / (d * E + C)), size=(d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward: gather embeddings for each position
            # emb_gathered: [bs, d, E]
            emb_gathered = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2)  # [bs, d, E]
            flat = emb_gathered.reshape(bs, d * E)                            # [bs, d*E]
            scores = flat @ W + b                                               # [bs, C]

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dscores = (probs / bs).astype(np.float32)                         # [bs, C]

            # Gradients
            dW = flat.T @ dscores + L2 * W                                     # [d*E, C]
            db = dscores.sum(axis=0)                                           # [C]
            dflat = dscores @ W.T                                              # [bs, d*E]
            demb_gathered = dflat.reshape(bs, d, E)                            # [bs, d, E]

            # Accumulate embedding gradients: demb[d, 256, E]
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], demb_gathered[:, j, :])
            demb += L2 * emb

            grads = [demb, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, C; then embeddings[d][256][E], W[d*E][C], b[C]
    magic = 0x454D424C  # "L B M E" little-endian
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Each input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W[d*E][C], b[C]. */
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
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *buf = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings into contiguous buffer
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = buf + (long)j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // Scores = b + buf @ W
        memcpy(scores, b, C * sizeof(float));
        for (int idx = 0; idx < d * E; idx++) {
            float val = buf[idx];
            if (val == 0.0f) continue;
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += val * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```