```hypothesis
We propose "label-emb-mlp": a shared byte-embedding (256×E) followed by position-specific linear projections to a hidden layer (H ReLU units) and a final linear output layer. This factorizes the large one-hot weight matrix (d×256×H) into a small embedding (256×E) and per-position matrices (d×E×H), drastically reducing parameters (E=8, H=64 → ~45k vs 524k for full one-hot MLP) which should improve generalization on the 24k training items. Inference cost is ~12k multiply-adds/item (between label-linear and label-mlp16). We expect dev accuracy >0.34 (beating label-linear's 0.3375) with cost <10k Ir/item. Failure: accuracy ≤0.3375 or cost ≥10k Ir/item.
```

```python
"""label-emb-mlp: shared byte embeddings + position-specific hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D424C ("LBME"), int32 d,E,H,C; float32 emb[256*E], W1[d*E*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8          # embedding dimension
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424C  # "LBME" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    emb = (rng.normal(0, 0.1, (256, E)) * math.sqrt(2.0 / 256)).astype(np.float32)
    W1 = (rng.normal(0, 0.1, (d, E, H)) * math.sqrt(2.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, 0.1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            h = b1[None, :]      # (B, H)
            for j in range(d):
                e = emb[xb[:, j]]        # (B, E)
                h += e @ W1[j]           # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2    # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Gradients
            grad_W2 = h_relu.T @ grad_logits          # (H, C)
            grad_b2 = grad_logits.sum(axis=0)         # (C,)
            grad_h = grad_logits @ W2.T               # (B, H)
            grad_h *= (h > 0)                         # ReLU derivative
            grad_b1 = grad_h.sum(axis=0)              # (H,)

            grad_W1 = np.zeros((d, E, H), np.float32)
            grad_emb = np.zeros((256, E), np.float32)
            for j in range(d):
                e = emb[xb[:, j]]                     # (B, E)
                grad_W1[j] = e.T @ grad_h             # (E, H)
                grad_e = grad_h @ W1[j].T             # (B, E)
                np.add.at(grad_emb, xb[:, j], grad_e)

            # L2 regularization
            grad_emb += L2 * emb
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
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
 * Each input record = 16 bytes. Output = 1 byte (class index).
 * Weights format: magic 0x454D424C, int32 d,E,H,C; float32 emb[256*E], W1[d*E*H], b1[H], W2[H*C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 20L
        + 4L * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = b1
        memcpy(h, b1, sizeof(float) * H);
        // for each position j: h += emb[xi[j]] @ W1[j]
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            const float *W1j = W1 + (long)j * E * H;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int kk = 0; kk < E; kk++) sum += e[kk] * W1j[(long)kk * H + u];
                h[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        // scores = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            scores[c] = sum;
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(scores);
    return 0;
}
```