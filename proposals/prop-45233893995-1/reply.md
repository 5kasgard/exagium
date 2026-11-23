```hypothesis
We propose "label-embed": learn a 4-d embedding for each of the 256 byte values per pixel, then a linear classifier on the concatenated 256-d vector. This captures non-linear per-pixel responses (pixel values 0..16 have complex class relationships) while inference stays fast (table lookups + one matvec). If dev accuracy does not exceed label-linear (0.9613), the extra parameters do not justify the cost.
```

```python
"""label-embed: per-feature byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x454D4244, d, E, C; float32 embeddings[d][256][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 4          # embedding size per feature
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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
    # embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    # linear layer
    W = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # concatenate embeddings
        feats = np.empty((batch, d * E), np.float32)
        for j in range(d):
            feats[:, j*E:(j+1)*E] = emb[j, Xb[:, j]]
        scores = feats @ W + b
        return feats, scores

    def loss_grad(feats, scores, yb):
        batch = feats.shape[0]
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(batch), yb] -= 1.0
        probs /= batch
        # grads
        gW = feats.T @ probs + L2 * W
        gb = probs.sum(axis=0)
        # embedding grads
        gemb = np.zeros_like(emb)
        gfeats = probs @ W.T  # (batch, d*E)
        for j in range(d):
            gj = gfeats[:, j*E:(j+1)*E]  # (batch, E)
            idx = Xb[:, j]
            np.add.at(gemb[j], idx, gj)
        gemb += L2 * emb
        return [gemb, gW, gb]

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            feats, scores = forward(Xb)
            grads = loss_grad(feats, scores, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d <= 0 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long emb_sz = (long)d * 256 * E;
    long W_sz = (long)d * E * C;
    long b_sz = C;
    long expected = 16 + 4L * (emb_sz + W_sz + b_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feats = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* gather embeddings */
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = feats + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }
        /* linear layer */
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < d * E; f++) sum += feats[f] * W[(long)f * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```