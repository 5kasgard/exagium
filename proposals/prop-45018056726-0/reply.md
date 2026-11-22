```hypothesis
Family: label-fm (Factorization Machine with 8‑dim embeddings).
Why: FMs capture pairwise feature interactions via the sum‑of‑squares trick, giving O(d·k) inference (d=14, k=8) while modeling non‑linearities that a linear model misses. The adult dataset has strong categorical interactions (e.g., education × occupation), which FMs handle well. The lookup‑table export mirrors the label‑linear pattern, so inference stays fast.
Failure: If dev accuracy ≤ 0.8524 (label‑linear) or instruction count exceeds label‑mlp16 (~2300 Ir/item), the model does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks (k=8).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports per‑byte lookup tables: magic 0x464D4C31, d, k, C; float32 T[d][256][1+k], bias."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices and compute Z (n x total_dim)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, total_dim = Z.shape

    # FM parameters
    w = np.zeros(total_dim, dtype=np.float32)
    V = (rng.normal(0, 0.01, (total_dim, K))).astype(np.float32)
    b = np.float32(0.0)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            B = len(yy)

            # Forward
            Vz = z @ V                         # (B, K)
            linear = z @ w + b                 # (B,)
            fm = 0.5 * (np.sum(Vz * Vz, axis=1) - np.sum((z * z) @ (V * V), axis=1))
            logits = linear + fm               # (B,)
            p = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            grad_s = (p - yy) / B              # (B,)

            # Gradients
            db = np.float32(grad_s.sum())
            dw = z.T @ grad_s + L2 * w
            # dV = z.T @ (grad_s[:,None] * Vz) - V * (grad_s @ (z*z))[:,None]
            dV = z.T @ (grad_s[:, None] * Vz) - V * (grad_s @ (z * z))[:, None] + L2 * V

            t += 1
            # Adam updates
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for param, grad, m, v in [
                (w, dw, m_w, v_w),
                (V, dV, m_V, v_V),
                (b, db, m_b, v_b),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Build per‑byte lookup tables T_j [256, 1+K]
    off = 0
    T_list = []
    for E in enc:
        D = E.shape[1]
        lin = (E @ w[off:off + D]).astype(np.float32)          # (256,)
        emb = (E @ V[off:off + D]).astype(np.float32)          # (256, K)
        T_list.append(np.concatenate([lin[:, None], emb], axis=1))  # (256, 1+K)
        off += D

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    tables = np.stack(T_list).astype("<f4").tobytes()  # (d, 256, 1+K)
    bias_bytes = np.array([b], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(header + tables + bias_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d=14). Output: one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + (long)d * 256 * (1 + K) * 4 + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);           // [d][256][1+K]
    const float bias = *(const float *)(T + (long)d * 256 * (1 + K));

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(K, sizeof(float));
    float *sum_sq_emb = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_sq_emb, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * (1 + K);
            score += t[0];                     // linear weight
            for (int f = 0; f < K; f++) {      // embedding
                float e = t[1 + f];
                sum_emb[f] += e;
                sum_sq_emb[f] += e * e;
            }
        }
        for (int f = 0; f < K; f++)
            score += 0.5f * (sum_emb[f] * sum_emb[f] - sum_sq_emb[f]);

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```