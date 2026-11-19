```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong feature interactions (e.g., education×occupation, marital-status×sex) that linear models miss. An FM with k=4 latent dimensions captures all pairwise interactions in O(d·k) time—similar to linear cost but more expressive. We reuse the card's per-byte lookup tables for both linear and latent weights. If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1500 (2× linear), the hypothesis fails.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-byte lookup tables (linear + k latent factors). Deterministic seed 1, single thread.
Exported: magic 0x464D3130, d, k, C; float32 T_lin[d][256][C], T_fm[d][256][k], b_lin[C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, L2_FM, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Build per-byte embedding tables
    T_lin = np.zeros((d, 256, C), dtype=np.float32)
    T_fm = np.zeros((d, 256, K), dtype=np.float32)
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            # Linear: z-scored values
            T_lin[j, :, :] = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            # FM: random init for each bin
            T_fm[j, :, :] = rng.normal(0, 0.01, (256, K)).astype(np.float32)
        else:
            # Categorical: one-hot for linear, random for FM
            vals = np.unique(X[:, j])
            for val in vals:
                T_lin[j, val, :] = 0.0
                T_fm[j, val, :] = rng.normal(0, 0.01, K).astype(np.float32)
            # Unseen (255) gets zero linear, random FM
            T_fm[j, 255, :] = rng.normal(0, 0.01, K).astype(np.float32)

    b_lin = np.zeros(C, dtype=np.float32)
    params = [T_lin, T_fm, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = math.ceil(n / BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    for step in range(max_steps):
        if step % steps_per_epoch == 0:
            order = rng.permutation(n)
        s = (step % steps_per_epoch) * BATCH
        idx = order[s:s + BATCH]
        xi, yy = X[idx], y[idx]
        bs = len(yy)

        # Gather embeddings
        lin_emb = T_lin[np.arange(d)[:, None], xi.T]  # (d, bs, C)
        fm_emb = T_fm[np.arange(d)[:, None], xi.T]    # (d, bs, K)

        # Linear logits
        lin_logits = lin_emb.sum(axis=0) + b_lin  # (bs, C)

        # FM interaction: 0.5 * (sum_v^2 - sum_v2)
        sum_v = fm_emb.sum(axis=0)               # (bs, K)
        sum_v2 = (fm_emb ** 2).sum(axis=0)       # (bs, K)
        fm_logits = 0.5 * ((sum_v ** 2) - sum_v2).sum(axis=1, keepdims=True)  # (bs, 1)

        logits = lin_logits + fm_logits

        # Softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(bs), yy] -= 1.0
        grad_logits = (probs / bs).astype(np.float32)  # (bs, C)

        # Gradients
        # b_lin
        grad_b_lin = grad_logits.sum(axis=0)

        # T_lin: for each (j, val), accumulate grad_logits over batch where xi==val
        grad_T_lin = np.zeros_like(T_lin)
        for j in range(d):
            vals = xi[:, j]
            np.add.at(grad_T_lin[j], vals, grad_logits)

        # T_fm: dL/dv = grad_logits[:, None] * (sum_v - v)
        # grad_logits has shape (bs, C), but fm_logits is (bs, 1) broadcast to C
        # Actually fm_logits is added to all classes equally, so dL/d(fm_logits) = grad_logits.sum(axis=1)
        grad_fm_scalar = grad_logits.sum(axis=1)  # (bs,)
        grad_T_fm = np.zeros_like(T_fm)
        for j in range(d):
            vals = xi[:, j]
            v_j = fm_emb[j]  # (bs, K)
            # d(fm_logits)/dv_j = (sum_v - v_j)
            dv = sum_v - v_j  # (bs, K)
            # Scale by grad_fm_scalar
            grad_v = grad_fm_scalar[:, None] * dv  # (bs, K)
            np.add.at(grad_T_fm[j], vals, grad_v)

        # L2 regularization
        grad_T_lin += L2 * T_lin
        grad_T_fm += L2_FM * T_fm
        grad_b_lin += L2 * b_lin

        grads = [grad_T_lin, grad_T_fm, grad_b_lin]
        t = step + 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, k, C; T_lin, T_fm, b_lin (all float32 little-endian)
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_fm.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax).
 * Weights format: magic 0x464D3130, d, k, C; float32 T_lin[d][256][C], T_fm[d][256][k], b_lin[C]. */
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
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_fm = T_lin + (long)d * 256 * C;
    const float *b_lin = T_fm + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) lin_acc[c] = b_lin[c];
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin_acc[c] += t[c];
        }

        // FM part
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v2[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *t = T_fm + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float v = t[u];
                sum_v[u] += v;
                sum_v2[u] += v * v;
            }
        }
        float fm_score = 0.0f;
        for (int u = 0; u < K; u++) {
            float diff = sum_v[u] * sum_v[u] - sum_v2[u];
            fm_score += 0.5f * diff;
        }

        // Add FM score to all classes (broadcast)
        for (int c = 0; c < C; c++) lin_acc[c] += fm_score;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (lin_acc[c] > lin_acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```