```hypothesis
Family: label-fm (Factorization Machine with rank-4 embeddings per column-value).
Why: Adult has strong pairwise interactions (education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions in O(d·rank) time (d=14, rank=4 → ~60 FLOPs/item) vs label-linear's O(d). The per-column 256-value embedding tables fit the quantile-bin/categorical encoding naturally.
Failure signal: Dev accuracy ≤ 0.8524 (label-linear) — interactions not useful or rank too small.
```
```python
"""Factorization Machine for tabular label tasks: per (column, value) linear weight + rank embedding.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: int32 magic 0x464D4C31, d, rank, C; float32 lin[d][256], emb[d][256][rank]."""
import json, math, struct, sys

import numpy as np

RANK = 4
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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    # Parameters: linear[d, 256], emb[d, 256, RANK]
    lin = (rng.uniform(-0.01, 0.01, (d, 256))).astype(np.float32)
    emb = (rng.uniform(-0.01, 0.01, (d, 256, RANK)) * math.sqrt(6.0 / (d * RANK))).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = xb.shape[0]

            # Forward: gather embeddings and linear weights
            # emb_j shape (bsz, d, RANK), lin_j shape (bsz, d)
            emb_j = emb[xb]          # (bsz, d, RANK) via advanced indexing
            lin_j = lin[:, xb.T].T   # (bsz, d)

            s_vec = emb_j.sum(axis=1)                 # (bsz, RANK)
            s_sq = (s_vec * s_vec).sum(axis=1)        # (bsz,)
            emb_sq = (emb_j * emb_j).sum(axis=(1, 2)) # (bsz,)
            inter = 0.5 * (s_sq - emb_sq)             # (bsz,)
            logit = lin_j.sum(axis=1) + inter         # (bsz,)

            # Gradient of log-loss: sigmoid(logit) - y
            p = 1.0 / (1.0 + np.exp(-logit))          # (bsz,)
            grad = (p - yb).astype(np.float32)        # (bsz,)

            # Backprop
            # dlogit/dlin_j = 1
            grad_lin_j = grad[:, None]                # (bsz, 1)
            # dlogit/demb_j = s_vec - emb_j  (broadcast)
            grad_emb_j = grad[:, None, None] * (s_vec[:, None, :] - emb_j)  # (bsz, d, RANK)

            # Accumulate gradients per (column, value)
            # Use np.add.at for scatter-add
            glin = np.zeros_like(lin)
            gemb = np.zeros_like(emb)
            for j in range(d):
                v = xb[:, j]
                np.add.at(glin[j], v, grad_lin_j[:, 0])
                np.add.at(gemb[j], v, grad_emb_j[:, j, :])

            glin /= bsz
            gemb /= bsz

            # L2 regularization
            glin += L2 * lin
            gemb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [(lin, glin, m_lin, v_lin), (emb, gemb, m_emb, v_emb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, RANK, C)
    out = header + lin.astype("<f4").tobytes() + emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0/1). FM with rank embeddings per (column,value). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], rank = hd[2], C = hd[3];
    if (C != 2 || rank < 1 || rank > 16 || d < 1 || d > 256) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * d * 256 * (1L + rank);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *emb = lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[16], sum_emb_sq;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = 0.0f;
        for (int r = 0; r < rank; r++) sum_emb[r] = 0.0f;
        sum_emb_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base = ((long)j * 256 + v);
            score += lin[base];
            const float *e = emb + base * rank;
            for (int r = 0; r < rank; r++) {
                float val = e[r];
                sum_emb[r] += val;
                sum_emb_sq += val * val;
            }
        }
        for (int r = 0; r < rank; r++) score += 0.5f * sum_emb[r] * sum_emb[r];
        score -= 0.5f * sum_emb_sq;

        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```