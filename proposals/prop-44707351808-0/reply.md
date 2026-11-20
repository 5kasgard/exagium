```hypothesis
Family: label-emb-mlp — per-feature embeddings (size 4) + one ReLU hidden layer (size 16).
Why: The 14 columns are all quantized to 256 bins; embeddings capture non-linear feature interactions more efficiently than one-hot + linear, while staying far smaller than a full one-hot MLP. Expect >0.8524 dev accuracy (beating label-linear) at ~1500 Ir/item (well below label-mlp16's 2325). Failure: dev accuracy ≤ 0.8524 or cost ≥ 2000 Ir/item.
```

```python
"""label-emb-mlp: per-feature embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weight format (little-endian):
  int32 magic=0x454D424D, d, E, H, C
  float32 emb[d][256][E]
  float32 W1[d*E][H], b1[H]
  float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

E = 4          # embedding size per feature
H = 16         # hidden layer size
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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Hidden layer: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Embedding lookup: (bs, d, E)
            emb_x = emb[np.arange(d)[None, :], xb[:, :, None], np.arange(E)[None, None, :]]  # (bs, d, E)
            flat = emb_x.reshape(bs, d * E)  # (bs, d*E)

            # Forward
            h = flat @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = (probs / bs).astype(np.float32)  # (bs, C)

            # Backprop
            grad_W2 = h_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T * (h > 0).astype(np.float32)
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Embedding gradients: (d, 256, E)
            grad_emb = np.zeros_like(emb)
            grad_flat = grad_h @ W1.T  # (bs, d*E)
            grad_emb_x = grad_flat.reshape(bs, d, E)
            np.add.at(grad_emb, (np.arange(d)[None, :], xb[:, :, None], np.arange(E)[None, None, :]), grad_emb_x)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t W2_sz = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + W1_sz + H + W2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + H;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, sizeof(float) * E);
        }

        // Hidden layer: h = ReLU(flat @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += flat[v] * W1[(size_t)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(size_t)u * C + c];
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