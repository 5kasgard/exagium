```hypothesis
Family: label-embed-linear. Learns a small embedding vector per feature value (256 per column) and a linear classifier on the concatenated embeddings. This captures similarity between categories (e.g., related occupations) that one-hot cannot, while staying extremely fast at inference (just table lookups + one matvec). If dev accuracy does not exceed label-linear (0.8524) with similar or lower Ir/item, the hypothesis fails.
```

```python
"""label-embed-linear: per-feature embeddings + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyper-parameters chosen on tuning split (last 20% of train.in)
SEED = 1
EMB_DIM = 4          # embedding dimension per feature
LR = 3e-3
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
SAMPLES = 600000     # same budget as label-linear

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
    d = X.shape[1]
    C = card["classes"]

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, tune_idx = idx[:split], idx[split:]

    # Embedding tables: (d, 256, EMB_DIM)
    E = (rng.normal(0, 0.01, (d, 256, EMB_DIM))).astype(np.float32)
    # Linear weights: (C, d*EMB_DIM), bias: (C,)
    W = np.zeros((C, d * EMB_DIM), np.float32)
    b = np.zeros(C, np.float32)

    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (B, d) uint8 -> (B, d*EMB_DIM) float32
        B = Xb.shape[0]
        emb = E[Xb]                      # (B, d, EMB_DIM)
        flat = emb.reshape(B, -1)        # (B, d*EMB_DIM)
        logits = flat @ W.T + b          # (B, C)
        return logits, flat

    def loss_grad(logits, yb):
        # softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return probs.astype(np.float32) / len(yb)

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / split))):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx_b = train_idx[order[s:s+BATCH]]
            Xb, yb = X[idx_b], y[idx_b]
            logits, flat = forward(Xb)
            g = loss_grad(logits, yb)            # (B, C)
            # Gradients
            gW = g.T @ flat + L2 * W             # (C, d*E)
            gb = g.sum(axis=0)                   # (C,)
            gflat = g @ W                        # (B, d*E)
            gemb = gflat.reshape(-1, d, EMB_DIM) # (B, d, E)
            # Accumulate per (feature, value)
            gE = np.zeros_like(E)
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(gE[j], vals, gemb[:, j, :])
            gE += L2 * E
            grads = [gE, gW, gb]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Final evaluation on tuning split (optional, but we already fixed hyperparams)
    # Pack weights: magic, d, EMB_DIM, C, then E (d*256*EMB_DIM), W (C*d*EMB_DIM), b (C)
    magic = 0x454D424C  # "LMBE" little-endian
    out = struct.pack("<4i", magic, d, EMB_DIM, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index.
 * Weights layout: magic(0x454D424C), d, E, C, embeddings[d][256][E], W[C][d*E], b[C] (all float32 LE). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d <= 0 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * d * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *Etab = (const float *)(hd + 4);
    const float *W = Etab + (long)d * 256 * E;
    const float *b = W + (long)C * d * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = Etab + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + (long)j * E, src, E * sizeof(float));
        }
        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *w = W + (long)c * d * E;
            for (int f = 0; f < d * E; f++) sum += emb_concat[f] * w[f];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```