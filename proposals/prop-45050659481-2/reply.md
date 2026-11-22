```hypothesis
Family: label-emb (embedding + linear). Learned per-position byte embeddings (dim=8) reduce 4096 one-hot features to 128 dense features, capturing character similarity. Expected to exceed label-linear accuracy (0.3375) with similar inference cost (128*65 vs 4096*65 multiply-adds). Failure: if dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear.
```

```python
"""Embedding + linear family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x454D424C, d, E, H, C; float32 emb[d][256][E], W[(d*E)][C], b[C] (H=0)."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension per position
H = 0          # hidden size (0 = linear)
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

    # parameters: embeddings[d][256, E], W[d*E, C], b[C]
    emb = [(rng.standard_normal((256, E), dtype=np.float32) * math.sqrt(1.0 / E)) for _ in range(d)]
    W = (rng.standard_normal((d * E, C), dtype=np.float32) * math.sqrt(2.0 / (d * E + C)))
    b = np.zeros(C, dtype=np.float32)
    params = emb + [W, b]

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
            bs = len(yb)

            # forward: gather embeddings
            z = np.zeros((bs, d * E), dtype=np.float32)
            for j in range(d):
                z[:, j*E:(j+1)*E] = emb[j][xb[:, j]]

            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # backward
            dW = z.T @ probs + L2 * W
            db = probs.sum(axis=0)
            dz = probs @ W.T

            demb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                demb[j][xb[:, j]] += dz[:, j*E:(j+1)*E]
            for j in range(d):
                demb[j] += L2 * emb[j]

            grads = demb + [dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    header = struct.pack("<5i", 0x454D424C, d, E, H, C)
    body = b"".join(p.astype("<f4").tobytes() for p in params)
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Embedding + linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 0 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * E);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(z + j * E, e, E * sizeof(float));
        }

        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int f = 0; f < d * E; f++) {
            float zv = z[f];
            if (zv == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) sc[c] += zv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```