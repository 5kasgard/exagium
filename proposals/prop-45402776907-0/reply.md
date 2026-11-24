```hypothesis
Family: label-embmlp (embedding + small MLP for tabular classification).
Why: Adult has 8 high-cardinality categorical features; one-hot (label-linear) wastes capacity and ignores category similarity.
Embeddings (E=8) compress each feature to a dense vector, preserving relationships. A 16-unit ReLU MLP on the concatenated
embeddings (14*8=112 dims) models interactions cheaply. This should beat label-linear's 0.8524 accuracy while staying
lighter than label-mlp16 (which uses one-hot + 16 hidden units).
Failure signal: dev accuracy ≤ 0.8524 or inference cost ≥ label-linear.
```

```python
"""label-embmlp: embedding + 1-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[(d*E)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension per feature
H = 16         # hidden units
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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: one per feature, 256 values each, E dims
    Emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # MLP weights
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Embedding lookup: (bs, d, E)
            emb = Emb[xb]                       # (bs, d, E)
            flat = emb.reshape(bs, d * E)       # (bs, d*E)

            # Forward
            h = flat @ W1 + b1                  # (bs, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2           # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backprop
            gW2 = h_relu.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh_relu = probs @ W2.T
            gh = gh_relu * (h > 0)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)

            # Embedding gradients: scatter-add
            gEmb = np.zeros_like(Emb)
            gh_emb = (gh @ W1.T).reshape(bs, d, E)   # (bs, d, E)
            np.add.at(gEmb, xb, gh_emb)
            gEmb += L2 * Emb

            grads = [gEmb, gW1, gb1, gW2, gb2]

            # Adam update
            t = 1  # we don't track global step across batches; use fixed bias correction approx
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                # bias correction (approximate with large t)
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights.bin
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D424C, d, E, H, C)
    out += Emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax).
 * Weights layout (float32 little-endian):
 *   int32 magic=0x454D424C, d, E, H, C
 *   float Emb[d][256][E]
 *   float W1[d*E][H], b1[H]
 *   float W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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