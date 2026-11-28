```hypothesis
Family: label-catfm (Linear + Categorical Factorization Machine, k=2).
Why: Adult has 8 categorical features with known pairwise interactions (e.g., education×occupation, marital-status×sex). A tiny FM on categorical features only adds ~4K params and ~2× linear inference cost, capturing 2-way interactions without the overhead of a full MLP or GBDT.
Success: dev accuracy > 0.8524 (label-linear) at < 1500 Ir/item. Failure: accuracy ≤ 0.8524 or Ir/item ≥ 3000.
```

```python
"""label-catfm: Linear + Categorical Factorization Machine (k=2) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports per-byte lookup tables for linear and FM embeddings."""
import json, math, struct, sys

import numpy as np

K_FM = 2
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
    types = card["input"]["types"]
    d = X.shape[1]
    assert d == card["input"]["bytes"]
    assert C == 2

    cat_indices = [j for j, t in enumerate(types) if t == "categorical"]
    num_cat = len(cat_indices)

    rng = np.random.default_rng(SEED)

    # Build linear encoding (same as label-linear)
    enc = []
    enc_dims = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
            enc_dims.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            enc_dims.append(len(vals))

    K = sum(enc_dims)

    # Linear weights
    W1 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b1 = np.zeros(C, np.float32)

    # FM embeddings for categorical features only: [num_cat, 256, K_FM]
    fm_emb = rng.uniform(-0.1, 0.1, (num_cat, 256, K_FM)).astype(np.float32)

    params = [W1, b1, fm_emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Linear features
            Z_parts = [E[xb[:, j]] for j, E in enumerate(enc)]
            Z = np.concatenate(Z_parts, axis=1)

            lin_logits = Z @ W1 + b1

            # FM part
            fm_sum = np.zeros((B, K_FM), dtype=np.float32)
            fm_sq_sum = np.zeros((B, K_FM), dtype=np.float32)
            for ci, j in enumerate(cat_indices):
                e = fm_emb[ci, xb[:, j]]
                fm_sum += e
                fm_sq_sum += e * e
            fm_logit = 0.5 * np.sum(fm_sum * fm_sum - fm_sq_sum, axis=1)

            logits = lin_logits.copy()
            logits[:, 0] -= fm_logit
            logits[:, 1] += fm_logit

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            d_logits = (probs / B).astype(np.float32)

            # Linear gradients
            d_W1 = Z.T @ d_logits + L2 * W1
            d_b1 = d_logits.sum(axis=0)

            # FM gradients
            d_fm_logit = d_logits[:, 1] - d_logits[:, 0]
            d_fm_sum = d_fm_logit[:, None] * fm_sum

            d_fm_emb = np.zeros_like(fm_emb)
            for ci, j in enumerate(cat_indices):
                e = fm_emb[ci, xb[:, j]]
                d_e = d_fm_sum - d_fm_logit[:, None] * e
                np.add.at(d_fm_emb[ci], xb[:, j], d_e)
            d_fm_emb += L2 * fm_emb

            grads = [d_W1, d_b1, d_fm_emb]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute linear lookup tables T[d][256][C]
    T = np.zeros((d, 256, C), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = enc_dims[j]
        T[j] = (E @ W1[off:off + dim]).astype(np.float32)
        off += dim

    # Write weights
    header = struct.pack("<5i", 0x464D4331, d, K_FM, C, num_cat)
    if num_cat:
        header += struct.pack(f"<{num_cat}i", *cat_indices)

    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if num_cat:
        out += fm_emb.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-catfm: Linear + Categorical FM (k=2) for label tasks.
 * ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x464D4331) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K_FM = hd[2], C = hd[3], num_cat = hd[4];
    if (C != 2 || K_FM != 2) { fprintf(stderr, "only C=2 K_FM=2 supported\n"); return 3; }

    int32_t *cat_indices