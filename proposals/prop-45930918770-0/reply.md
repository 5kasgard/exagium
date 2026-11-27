```hypothesis
Factorization Machine (label-fm) with per-feature-value embeddings (dimension 8) captures pairwise feature interactions efficiently. On this tabular task with mixed ordinal/categorical columns, FM should outperform the linear baseline (0.8524) by modeling interactions while keeping inference fast (O(d * features) per item). If dev accuracy ≤ 0.8524 or instructions/item exceed label-linear by >20%, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic, d, num_features, num_classes, num_values; then per-class bias, linear[features][values], emb[features][values][d]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 8              # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50

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
    n, d = X.shape
    C = card["classes"]
    V = 256  # values per feature (0-255)

    rng = np.random.default_rng(SEED)
    # Parameters per class: bias, linear[d][V], emb[d][V][D]
    # Initialize linear weights small, embeddings normal(0, 0.01)
    linear = rng.normal(0, 0.01, size=(C, d, V)).astype(np.float32)
    emb = rng.normal(0, 0.01, size=(C, d, V, D)).astype(np.float32)
    bias = np.zeros((C,), dtype=np.float32)

    params = [bias, linear, emb]
    shapes = [p.shape for p in params]
    # Adam state
    state = [(np.zeros_like(p, dtype=np.float32), np.zeros_like(p, dtype=np.float32)) for p in params]
    t = 0

    def forward_batch(xb):  # xb: (B, d) uint8
        B = xb.shape[0]
        logits = np.zeros((B, C), dtype=np.float32)
        for c in range(C):
            logits[:, c] = bias[c]
            # Linear term
            for j in range(d):
                logits[:, c] += linear[c, j, xb[:, j]]
            # FM interaction term
            # emb_sum: (B, D)
            emb_sum = np.zeros((B, D), dtype=np.float32)
            emb_sq_sum = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                e = emb[c, j, xb[:, j]]  # (B, D)
                emb_sum += e
                emb_sq_sum += e * e
            inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
            logits[:, c] += inter
        return logits

    def backward_batch(xb, yb, logits, grads):  # grads: list of (gbias, glinear, gemb)
        B = xb.shape[0]
        # Softmax cross-entropy gradient
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        probs /= B  # (B, C)

        for c in range(C):
            g = probs[:, c]  # (B,)
            # bias gradient
            grads[c][0] += g.sum()
            # linear gradient
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grads[c][1][j], vals, g)
            # embedding gradient
            # For each feature j, we need d/emb_sum per sample
            # Recompute emb_sum per class
            emb_sum = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                e = emb[c, j, xb[:, j]]
                emb_sum += e
            for j in range(d):
                e = emb[c, j, xb[:, j]]  # (B, D)
                # grad for emb[c,j,val] = g[:,None] * (emb_sum - e)
                ge = g[:, None] * (emb_sum - e)  # (B, D)
                vals = xb[:, j]
                # Accumulate per value
                for val in range(V):
                    mask = (vals == val)
                    if np.any(mask):
                        grads[c][2][j, val] += ge[mask].sum(axis=0)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            logits = forward_batch(xb)
            # Gradients
            grads = []
            for c in range(C):
                gbias = np.zeros_like(bias[c])
                glinear = np.zeros_like(linear[c])
                gemb = np.zeros_like(emb[c])
                grads.append([gbias, glinear, gemb])
            backward_batch(xb, yb, logits, grads)
            # L2 regularization on linear and emb (not bias)
            for c in range(C):
                grads[c][1] += L2 * linear[c]
                grads[c][2] += L2 * emb[c]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m = m * 0.9 + 0.1 * gr
                v = v * 0.999 + 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                state[i] = (m, v)

    # Write weights
    magic = 0x464D3132  # 'FM12'
    # Flatten order: bias[C], linear[C][d][V], emb[C][d][V][D]
    total_floats = C * (1 + d * V + d * V * D)
    expected_bytes = 5 * 4 + total_floats * 4  # 5 int32 header
    out = struct.pack("<5i", magic, D, d, C, V)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    assert len(out) == expected_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 20 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int D = hd[1], d = hd[2], C = hd[3], V = hd[4];
    if (C != 2 || V != 256 || D < 1 || D > 64 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    // Sizes
    size_t bias_sz = C;
    size_t linear_sz = C * d * V;
    size_t emb_sz = C * d * V * D;
    size_t total_floats = bias_sz + linear_sz + emb_sz;
    if (nw != 20 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 5);
    const float *bias = p; p += bias_sz;
    const float *linear = p; p += linear_sz;
    const float *emb = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *emb_sq_sum = malloc(sizeof(float) * D);
    float logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = bias[c];
            // Linear term
            for (int j = 0; j < d; j++) {
                logits[c] += linear[((c * d + j) * V) + xi[j]];
            }
            // FM interaction
            memset(emb_sum, 0, D * sizeof(float));
            memset(emb_sq_sum, 0, D * sizeof(float));
            for (int j = 0; j < d; j++) {
                const float *e = emb + (((c * d + j) * V) + xi[j]) * D;
                for (int f = 0; f < D; f++) {
                    float val = e[f];
                    emb_sum[f] += val;
                    emb_sq_sum[f] += val * val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < D; f++) {
                inter += emb_sum[f] * emb_sum[f] - emb_sq_sum[f];
            }
            logits[c] += 0.5f * inter;
        }

        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```