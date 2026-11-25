```hypothesis
Family: label-embed-mlp. A shared token embedding (256 x E) plus positional embeddings (16 x E) fed into a one-hidden-layer MLP (ReLU) with hidden size H. This captures positional n-gram statistics at lower parameter cost than full one-hot linear (16*256=4096 vs 16*E+H*(16*E)+H*65). With E=16, H=64 it has ~20k parameters vs linear's 266k, so it should be faster and more accurate on this small-context character task. If dev accuracy ≤ 0.3375 or instructions/item ≥ 7359, the hypothesis fails.
```
```python
"""label-embed-mlp: shared token embedding + positional embedding + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d=16, E, H, C=65
  float32 token_emb[256][E]
  float32 pos_emb[16][E]
  float32 W1[16*E][H], b1[H]
  float32 W2[H][C], b2[C]
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 16          # embedding dimension
H = 64          # hidden size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424C  # "LMBE" little-endian

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
    assert d == 16 and C == 65

    X_bytes = b"".join(records(sys.argv[2]))
    n = len(X_bytes) // d
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    assert len(y) == n

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    token_emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    pos_emb   = (rng.uniform(-1, 1, (d, E))     * math.sqrt(6.0 / d)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [token_emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = len(yb)

            # Forward
            # token_emb[xb] -> (B, d, E), add pos_emb -> (B, d, E)
            emb = token_emb[xb] + pos_emb          # (B, d, E)
            flat = emb.reshape(B, d * E)           # (B, d*E)
            h = flat @ W1 + b1                     # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2              # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Backward
            dW2 = h_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh_relu = dlogits @ W2.T
            dh = dh_relu * (h > 0)
            dW1 = flat.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            demb_flat = dh @ W1.T                       # (B, d*E)
            demb = demb_flat.reshape(B, d, E)           # (B, d, E)

            # Gradient for token_emb: scatter-add
            dtoken_emb = np.zeros_like(token_emb)
            np.add.at(dtoken_emb, xb.ravel(), demb.reshape(-1, E))
            dtoken_emb += L2 * token_emb

            # Gradient for pos_emb: sum over batch
            dpos_emb = demb.sum(axis=0) + L2 * pos_emb

            grads = [dtoken_emb, dpos_emb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, E, H, C)
    out += token_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d=16, E, H, C=65
 *   float32 token_emb[256][E]
 *   float32 pos_emb[16][E]
 *   float32 W1[16*E][H], b1[H]
 *   float32 W2[H][C], b2[C]
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20L
        + 4L * 256 * E
        + 4L * d * E
        + 4L * d * E * H + 4L * H
        + 4L * H * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *token_emb = (const float *)(hd + 5);
    const float *pos_emb   = token_emb + 256 * E;
    const float *W1        = pos_emb + d * E;
    const float *b1        = W1 + d * E * H;
    const float *W2        = b1 + H;
    const float *b2        = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embed: token_emb[xi[j]] + pos_emb[j]
        for (int j = 0; j < d; j++) {
            const float *te = token_emb + xi[j] * E;
            const float *pe = pos_emb + j * E;
            float *out = emb + j * E;
            for (int e = 0; e < E; e++) out[e] = te[e] + pe[e];
        }

        // hidden layer: h = ReLU(emb * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += emb[v] * W1[v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: scores = h * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            scores[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```