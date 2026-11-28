```hypothesis
Family: label-embed-mlp (shared byte embeddings + positional embeddings + 1-hidden-layer MLP).
Why: The 16-byte context is a sequence; shared embeddings capture byte identity across positions while positional embeddings retain order. A small MLP on the concatenated embeddings models interactions more expressively than a linear model, yet stays tiny (≈14k params) for fast inference. Beats label-linear by learning distributed representations; fails if dev accuracy ≤ 0.34 or latency > 2× label-linear.
```

```python
"""label-embed-mlp: shared byte embeddings + positional embeddings + 1-hidden-layer ReLU MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Weights format (little-endian):
  int32 magic=0x454D4244 ('DBME'), d=16, E, H, C
  float32 shared_emb[256*E], pos_emb[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 16          # embedding dimension
H = 64          # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20

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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    shared_emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    pos_emb    = (rng.normal(0, 0.01, (d, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [shared_emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            # xb: (bs, d) uint8
            emb = shared_emb[xb] + pos_emb  # (bs, d, E)
            flat = emb.reshape(bs, d * E)   # (bs, d*E)
            h = np.maximum(flat @ W1 + b1, 0)  # (bs, H)
            logits = h @ W2 + b2                # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs  # (bs, C)

            # Backward
            grad_W2 = h.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_h = probs @ W2.T
            grad_h[h == 0] = 0  # ReLU derivative
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_flat = grad_h @ W1.T  # (bs, d*E)
            grad_emb = grad_flat.reshape(bs, d, E)  # (bs, d, E)

            # Accumulate gradients for shared_emb and pos_emb
            grad_shared = np.zeros_like(shared_emb)
            np.add.at(grad_shared, xb, grad_emb)
            grad_shared += L2 * shared_emb
            grad_pos = grad_emb.sum(axis=0) + L2 * pos_emb

            grads = [grad_shared, grad_pos, grad_W1, grad_b1, grad_W2, grad_b2]

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
    magic = 0x454D4244  # 'DBME'
    header = struct.pack("<5i", magic, d, E, H, C)
    blob = b"".join(p.astype("<f4").tobytes() for p in params)
    open(sys.argv[4], "wb").write(header + blob)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t expected = 20 +
        (size_t)256 * E * 4 +
        (size_t)d * E * 4 +
        (size_t)d * E * H * 4 +
        (size_t)H * 4 +
        (size_t)H * C * 4 +
        (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *shared_emb = p; p += 256 * E;
    const float *pos_emb    = p; p += d * E;
    const float *W1 = p; p += d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embeddings: shared_emb[byte] + pos_emb[position]
        for (int j = 0; j < d; j++) {
            const float *se = shared_emb + (size_t)xi[j] * E;
            const float *pe = pos_emb + (size_t)j * E;
            float *out = emb_concat + j * E;
            for (int e = 0; e < E; e++) out[e] = se[e] + pe[e];
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_concat[k] * W1[(size_t)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_concat); free(hidden); free(logits);
    return 0;
}
```