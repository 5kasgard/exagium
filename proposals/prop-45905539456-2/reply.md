```hypothesis
Family: label-embmlp (shared byte embedding + positional encoding + one hidden ReLU layer).
Why: Character-level sequence modeling benefits from distributed representations. A shared 16-dim embedding for each of 256 byte values (4096 params) plus 16 positional embeddings (256 params) compresses the 16×256 one-hot input to 256 dense features. A 64-unit ReLU layer (16k params) then captures cross-position interactions before the 65-class output (4k params). Total ~25k parameters vs. linear's 266k, so it should generalize better and run faster. Beats label-linear (0.3375) if dev accuracy > 0.34; otherwise the bottleneck is too aggressive.
```

```python
"""label-embmlp: shared byte embedding + positional encoding + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic 0x454D424D ('EMBM')
  int32  d (input bytes, 16)
  int32  E (embedding dim, 16)
  int32  H (hidden units, 64)
  int32  C (classes, 65)
  float32 emb[256][E]
  float32 pos_emb[d][E]
  float32 W1[d*E][H]
  float32 b1[H]
  float32 W2[H][C]
  float32 b2[C]
"""
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: 256 x E, He init for ReLU (fan_in=1)
    emb = (rng.normal(0, math.sqrt(2.0), (256, E))).astype(np.float32)
    # Positional embeddings: d x E, small init
    pos_emb = (rng.normal(0, 0.01, (d, E))).astype(np.float32)
    # W1: (d*E) x H, He init
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C, Xavier init (fan_in=H, fan_out=C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, pos_emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, d, E)
            x_emb = emb[xb]                   # (B, d, E)
            x_emb += pos_emb                  # add positional embeddings (broadcast)
            x_flat = x_emb.reshape(B, -1)     # (B, d*E)
            h_pre = x_flat @ W1 + b1          # (B, H)
            h = np.maximum(h_pre, 0)          # ReLU
            logits = h @ W2 + b2              # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h[h_pre <= 0] = 0.0          # ReLU derivative
            grad_W1 = x_flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Gradient for embeddings: need to scatter-add from grad_h @ W1.T reshaped to (B, d, E)
            grad_x_flat = grad_h @ W1.T       # (B, d*E)
            grad_x_emb = grad_x_flat.reshape(B, d, E)   # (B, d, E)
            # Positional embeddings gradient: sum over batch
            grad_pos_emb = grad_x_emb.sum(axis=0)
            # Embedding gradient: scatter-add per byte value
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, xb, grad_x_emb)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_pos_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * (gr * gr)
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Pack weights
    # Order: emb, pos_emb, W1, b1, W2, b2
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        pos_emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], E = (int)hd[2], H = (int)hd[3], C = (int)hd[4];
    if (d != 16 || E != 16 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    size_t expected_floats = 256*E + d*E + (size_t)d*E*H + H + (size_t)H*C + C;
    size_t expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected_bytes); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *pos_emb = p; p += d * E;
    const float *W1 = p; p += (size_t)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (size_t)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_flat = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + positional encoding
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            const float *pe = pos_emb + j * E;
            float *dst = x_flat + j * E;
            for (int kk = 0; kk < E; kk++) dst[kk] = e[kk] + pe[kk];
        }

        // Hidden layer: h = ReLU(x_flat @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += x_flat[v] * W1[v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_flat); free(h); free(logits);
    if (x) free(x);
    return 0;
}
```