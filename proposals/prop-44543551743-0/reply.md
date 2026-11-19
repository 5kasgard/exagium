```hypothesis
Family: label-emb-mlp (entity-embedding MLP). Treats all 14 columns as categorical (256 values each), learns a small embedding per column/value, then a 2-layer ReLU MLP. Embeddings capture non-linear interactions that one-hot+linear misses, while staying far smaller than a full one-hot MLP. Should beat label-linear (0.8524) and approach label-mlp16 (0.8591) with lower inference cost (no giant one-hot expansion). If dev accuracy ≤ 0.8525 or Ir/item ≥ 2000, the idea fails.
```

```python
"""label-emb-mlp: entity embeddings + 2-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format (little-endian):
  magic(0x454D424D), d, E, H, C (int32)
  embeddings[d][256][E] (float32)
  W1[d*E][H] (float32), b1[H] (float32)
  W2[H][C] (float32), b2[C] (float32)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (would be chosen on tuning split)
E = 4          # embedding size per feature
H = 32         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424D  # 'EMBM'

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

    # Embeddings: (d, 256, E)
    emb = (rng.normal(0, 1/math.sqrt(E), size=(d, 256, E))).astype(np.float32)

    # MLP weights
    W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # Embedding lookup: (bs, d, E)
            emb_x = emb[xb]                       # (bs, d, E)
            flat = emb_x.reshape(bs, d*E)         # (bs, d*E)
            h_pre = flat @ W1 + b1                # (bs, H)
            h = np.maximum(h_pre, 0)              # ReLU
            logits = h @ W2 + b2                  # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backprop
            grad_W2 = h.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_h = probs @ W2.T
            grad_h[h_pre <= 0] = 0.0              # ReLU derivative
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            grad_flat = grad_h @ W1.T              # (bs, d*E)
            grad_emb_x = grad_flat.reshape(bs, d, E)
            np.add.at(grad_emb, (xb[:,:,None], np.arange(E)), grad_emb_x.transpose(1,0,2))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1-beta1) * gr
                v *= beta2; v += (1-beta2) * gr * gr
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index.
 * Weights layout (little-endian):
 *   magic(0x454D424D), d, E, H, C (int32)
 *   embeddings[d][256][E] (float32)
 *   W1[d*E][H] (float32), b1[H] (float32)
 *   W2[H][C] (float32), b2[C] (float32)
 */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz  = (size_t)d * E * H;
    size_t W2_sz  = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + W1_sz + H + W2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1  = emb + emb_sz;
    const float *b1  = W1 + W1_sz;
    const float *W2  = b1 + H;
    const float *b2  = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h    = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + flatten
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, src, E * sizeof(float));
        }

        // Hidden layer: h = ReLU(flat @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d*E; v++) sum += flat[v] * W1[(size_t)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(size_t)u * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```