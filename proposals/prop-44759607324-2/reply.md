```hypothesis
Family: label-mlp2h (two-hidden-layer MLP with per-position byte embeddings).
Reason: The tinyshakespeare task is character-level language modeling; a deeper network can capture higher-order n-gram interactions that a single linear layer or one-hidden-layer MLP misses. The first layer is still reduced to embedding lookups and sums, so inference stays fast (≈10k Ir/item). If dev accuracy does not exceed label-linear (0.3375) by ≥0.5% absolute, the extra depth is not justified.
```python
"""Two-hidden-layer MLP with per-position byte embeddings (family label-mlp2h).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: 16 positional embedding tables (256 x W) summed -> ReLU(H1) -> ReLU(H2) -> linear(C).
Hyperparameters chosen on tuning split (last 20% of train.in): W=64, H1=256, H2=128, lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exported weights format (little-endian):
  int32 magic=0x54424C32, d, H1, H2, C
  float32 T[d][256][H1]   (precomputed E @ W1)
  float32 b1[H1]
  float32 W2[H1][H2]
  float32 b2[H2]
  float32 W3[H2][C]
  float32 b3[C]
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on validation split)
W, H1, H2 = 64, 256, 128
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
    C = card["classes"]
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    # Build per-position one-hot encoders (only for values seen in training)
    enc = []
    for j in range(d):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)

    # Total one-hot dimension
    total_dim = sum(E.shape[1] for E in enc)

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (total_dim, H1)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute concatenated one-hot matrix Z (n x total_dim)
    n = X.shape[0]
    Z = np.empty((n, total_dim), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        sz = E.shape[1]
        Z[:, off:off+sz] = E[X[:, j]]
        off += sz

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]

            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g3 = (probs / len(yy)).astype(np.float32)

            # Backprop
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(axis=0)

            gh2 = g3 @ W3.T
            gh2 *= (a2 > 0)
            grad_W2 = h1.T @ gh2 + L2 * W2
            grad_b2 = gh2.sum(axis=0)

            gh1 = gh2 @ W2.T
            gh1 *= (a1 > 0)
            grad_W1 = z.T @ gh1 + L2 * W1
            grad_b1 = gh1.sum(axis=0)

            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute per-position lookup tables T[j][byte][H1] = E_j @ W1_j
    T_list = []
    off = 0
    for E in enc:
        sz = E.shape[1]
        T_list.append((E @ W1[off:off+sz]).astype(np.float32))
        off += sz

    # Write weights file
    magic = 0x54424C32  # "TBL2"
    header = struct.pack("<5i", magic, d, H1, H2, C)
    body = b"".join(arr.tobytes() for arr in T_list)
    body += b1.tobytes()
    body += W2.tobytes()
    body += b2.tobytes()
    body += W3.tobytes()
    body += b3.tobytes()

    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-mlp2h inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x54424C32, d, H1, H2, C
 *   float32 T[d][256][H1]
 *   float32 b1[H1]
 *   float32 W2[H1][H2]
 *   float32 b2[H2]
 *   float32 W3[H2][C]
 *   float32 b3[C]
 */
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
    if (nw < 20 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Layer 1: sum embeddings + bias, then ReLU
        for (int u = 0; u < H1; u++) h1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        // Layer 2: h1 @ W2 + b2, then ReLU
        for (int v = 0; v < H2; v++) h2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) h2[v] += hu * w[v];
        }
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        // Output layer: h2 @ W3 + b3
        for (int c = 0; c < C; c++) logits[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```