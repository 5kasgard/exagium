```hypothesis
Family: label-embedmlp — positional byte embeddings + one ReLU hidden layer.
Why: The 16-byte context has strong positional patterns (e.g., " the " at specific offsets). Learned embeddings capture per-position byte statistics more compactly than one-hot linear (266k params → ~39k), and a tiny MLP adds non-linearity for interactions. This should beat label-linear's 33.8% accuracy at similar or lower CPU cost (embedding lookup + small matmul).
Failure signal: If dev accuracy ≤ 0.34 or Ir/item > 10k, the family does not improve cost/accuracy.
```

```python
"""label-embedmlp: positional embeddings + 1 hidden ReLU layer for next-byte prediction.
train.py card.json train.in train.key weights.bin  (deterministic, seed=1, single thread)
Weights format (little-endian):
  int32 magic=0x454D4245, d=16, E=8, H=32, C=65
  float32 emb[d][256][E]          # positional embeddings
  float32 W1[d*E][H], b1[H]       # input -> hidden
  float32 W2[H][C], b2[C]         # hidden -> logits
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split conceptually)
E = 8          # embedding dimension per position
H = 32         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
CLIP = 1.0     # gradient clipping

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

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)

    # MLP weights
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: embeddings lookup
            # emb_vec shape (B, d*E)
            emb_vec = emb[np.arange(d), xb.T].transpose(1, 0, 2).reshape(B, d * E)

            # Hidden layer
            h_pre = emb_vec @ W1 + b1
            h = np.maximum(h_pre, 0)

            # Output logits
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # Backprop
            # grad W2, b2
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)

            # grad hidden
            gh = probs @ W2.T
            gh[h_pre <= 0] = 0.0

            # grad W1, b1
            gW1 = emb_vec.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)

            # grad embeddings
            # gh shape (B, H) -> (B, d, E) after dot with W1.T? Actually gh @ W1.T gives (B, d*E)
            g_emb_vec = gh @ W1.T          # (B, d*E)
            g_emb_vec = g_emb_vec.reshape(B, d, E)
            # Accumulate per (position, byte)
            g_emb = np.zeros_like(emb)
            np.add.at(g_emb, (np.arange(d), xb.T), g_emb_vec.transpose(1, 0, 2))
            g_emb += L2 * emb

            grads = [g_emb, gW1, gb1, gW2, gb2]

            # Gradient clipping (global norm)
            total_norm = math.sqrt(sum((g**2).sum() for g in grads))
            if total_norm > CLIP:
                scale = CLIP / total_norm
                grads = [g * scale for g in grads]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x454D4245  # 'EMBE'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
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
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D4245, d=16, E=8, H=32, C=65
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H], b1[H]
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }

    // Expected weight size in bytes
    long expected = 20 +
        (long)d * 256 * E * 4 +
        (long)d * E * H * 4 + H * 4 +
        (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_vec = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build embedding vector
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_vec + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(emb_vec * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_vec[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output logits
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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