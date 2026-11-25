```hypothesis
Family: label-fm (Factorization Machine, rank-8 embeddings). Why: Adult has strong pairwise interactions (education×occupation, marital-status×relationship) that linear models miss; FM captures them at O(d·k) inference cost (k=8), between linear and MLP in accuracy/latency. Failure signal: dev accuracy ≤ 0.8524 (linear) or Ir/item ≥ 2000 (mlp16 territory).
```

```python
"""Factorization Machine (rank-k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, k, C; float32 W[d][256], V[d][256][k], b[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, L2V, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 1e-5, 256, 400000, 80


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

    # One-hot encoding per feature (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k_enc = Z.shape

    # FM parameters: linear W[k_enc], embeddings V[k_enc, K], bias b[C]
    scale = math.sqrt(6.0 / (k_enc + K))
    W = (rng.uniform(-1, 1, (k_enc, C)) * scale).astype(np.float32)
    V = (rng.uniform(-1, 1, (k_enc, K)) * scale).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = z.shape[0]

            # Forward: linear + FM interaction
            lin = z @ W + b                                     # [m, C]
            zv = z @ V                                           # [m, K]
            inter = 0.5 * ((zv ** 2).sum(1, keepdims=True) - (z ** 2) @ (V ** 2))  # [m, 1]
            logits = lin + inter                                 # [m, C]

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_logits = probs / m                              # [m, C]

            # Gradients
            grad_W = z.T @ grad_logits + L2 * W                  # [k_enc, C]
            grad_b = grad_logits.sum(0)                          # [C]

            # FM interaction gradient w.r.t. V
            # dL/dV = z.T @ (grad_logits.sum(1, keepdims=True) * zv) - (z**2).T @ (grad_logits.sum(1, keepdims=True) * V)
            g_sum = grad_logits.sum(1, keepdims=True)            # [m, 1]
            grad_V = (z.T @ (g_sum * zv)) - ((z ** 2).T @ (g_sum * V)) + L2V * V  # [k_enc, K]

            grads = [grad_W, grad_V, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                m_hat = m1 / (1 - 0.9 ** t)
                v_hat = v1 / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export per-byte lookup tables for fast inference (same pattern as label-linear)
    # We need W_lookup[d][256][C], V_lookup[d][256][K], b[C]
    W_lookup = np.zeros((d, 256, C), np.float32)
    V_lookup = np.zeros((d, 256, K), np.float32)
    off = 0
    for j, E in enumerate(enc):
        nvals = E.shape[1]
        W_lookup[j] = E @ W[off:off + nvals]
        V_lookup[j] = E @ V[off:off + nvals]
        off += nvals

    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lookup.astype("<f4").tobytes()
    out += V_lookup.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (rank-K) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *V = W + (long)d * 256 * C;
    const float *b = V + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *zv = malloc(sizeof(float) * K);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear term + bias
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *w = W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += w[c];
        }

        // FM interaction: 0.5 * ( (sum v_i)^2 - sum v_i^2 )
        memset(zv, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) zv[u] += v[u];
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += zv[u] * zv[u];
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) inter -= v[u] * v[u];
        }
        inter *= 0.5f;
        for (int c = 0; c < C; c++) sc[c] += inter;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```