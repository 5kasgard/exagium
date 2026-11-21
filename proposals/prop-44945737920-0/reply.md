```hypothesis
Family: label-emb-mlp — embedding MLP with per-feature lookup tables.
Why: Adult has 8 categorical features (256 values each) and 6 ordinal features (quantile-binned). A linear model treats each value independently; an embedding layer learns dense representations that capture similarity between categories and quantile bins, then a small ReLU hidden layer models interactions. This adds minimal inference cost (one gather + two matvecs per record) while increasing capacity.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ 2× label-linear.
```

```python
"""label-emb-mlp: embedding + one ReLU layer for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic=0x454D424D ('EMBM'), d, E, H, C (int32)
  embed[d][256][E] (float32)  -- per-feature, per-value embedding
  W1[d*E][H] (float32), b1[H] (float32)
  W2[H][C] (float32), b2[C] (float32)"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 4          # embedding dimension per feature
H = 32         # hidden layer size
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

    # Embeddings: (d, 256, E)
    embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Z):
        # Z: (batch, d) uint8
        # Embedding lookup: (batch, d, E)
        emb = embed[np.arange(d), Z.T].transpose(1, 0, 2)  # (batch, d, E)
        flat = emb.reshape(Z.shape[0], d * E)               # (batch, d*E)
        h = flat @ W1 + b1
        h = np.maximum(h, 0, dtype=np.float32)
        logits = h @ W2 + b2
        return logits, h, flat

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Z, yy = X[idx], y[idx]
            logits, h, flat = forward(Z)
            g = softmax_grad(logits, yy)                # (batch, C)

            # Output layer grads
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T                                # (batch, H)
            gh *= (h > 0)                                # ReLU derivative

            # Hidden layer grads
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gflat = gh @ W1.T                            # (batch, d*E)

            # Embedding grads: scatter-add
            gemb = np.zeros_like(embed)
            gflat_3d = gflat.reshape(Z.shape[0], d, E)   # (batch, d, E)
            np.add.at(gemb, (np.arange(d), Z.T), gflat_3d.transpose(1, 0, 2))
            gemb += L2 * embed

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += embed.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index (0..C-1).
 * Weights layout (little-endian):
 *   int32 magic=0x454D424D, d, E, H, C
 *   float embed[d][256][E]
 *   float W1[d*E][H], b1[H]
 *   float W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H + 4L * H
        + 4L * H * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat
        for (int j = 0; j < d; j++) {
            const float *src = embed + ((long)j * 256 + xi[j]) * E;
            float *dst = flat + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }

        // Hidden layer: h = ReLU(flat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += flat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```