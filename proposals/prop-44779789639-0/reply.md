```hypothesis
Family: label-fm (Factorization Machine with per-field latent vectors).
Why: Adult has strong feature interactions (education×occupation, marital-status×sex) that linear models miss. FM adds pairwise interactions via low-rank latent vectors (rank k=4) shared across classes, keeping inference O(d²k) ≈ 400 ops — far cheaper than MLP. The per-field (original column) latent vectors match the tabular structure.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear), pairwise interactions aren't sufficient or rank 4 is too small.
```

```python
"""Factorization Machine for tabular label tasks: linear per-class weights + shared pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin
Exports: magic 0x464D3131, d, k, C; float32 T[d][256][C] (linear), V[d][256][k] (FM vectors), bias[C]."""
import json, math, struct, sys

import numpy as np

FM_RANK = 4
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
    rng = np.random.default_rng(SEED)

    enc = []
    enc_dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        enc_dims.append(E.shape[1])

    total_dim = sum(enc_dims)
    offs = np.cumsum([0] + enc_dims)

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    W_lin = (rng.uniform(-1, 1, (total_dim, C)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
    W_fm = (rng.uniform(-1, 1, (total_dim, FM_RANK)) * math.sqrt(6.0 / (total_dim * FM_RANK))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [W_lin, W_fm, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, y_idx):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs / len(y_idx)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]

            linear = z @ W_lin + bias
            sum_vz = z @ W_fm
            sum_vz_sq = (z * z) @ (W_fm * W_fm)
            interaction = 0.5 * np.sum(sum_vz * sum_vz - sum_vz_sq, axis=1, keepdims=True)

            logits = linear + interaction
            grad_logits = softmax_grad(logits, yy)

            grad_W_lin = z.T @ grad_logits + L2 * W_lin
            grad_bias = grad_logits.sum(axis=0)

            grad_interaction = grad_logits.sum(axis=1)
            factor = grad_interaction[:, None] * z
            grad_W_fm = np.zeros_like(W_fm)
            for f in range(FM_RANK):
                term = sum_vz[:, f:f+1] - z * W_fm[:, f:f+1]
                grad_W_fm[:, f] = factor.T @ term
            grad_W_fm += L2 * W_fm

            grads = [grad_W_lin, grad_W_fm, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    T_list, V_list = [], []
    for j, E in enumerate(enc):
        off, dim = offs[j], enc_dims[j]
        T_list.append(E @ W_lin[off:off+dim])
        V_list.append(E @ W_fm[off:off+dim])

    T_arr = np.stack(T_list).astype("<f4")
    V_arr = np.stack(V_list).astype("<f4")
    bias_arr = bias.astype("<f4")

    out = struct.pack("<4i", 0x464D3131, X.shape[1], FM_RANK, C)
    out += T_arr.tobytes() + V_arr.tobytes() + bias_arr.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Weights: magic 0x464D3131, d, k, C; float32 T[d][256][C], V[d][256][k], bias[C].
 * Output: one byte per record (class index). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k < 1 || k > 32 || d != 14) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * k + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *V = T + (long)d * 256 * C;
    const float *bias = V + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *latent = malloc(sizeof(float) * d * k);
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        scores[0] = bias[0]; scores[1] = bias[1];
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *tp = T + ((long)j * 256 + val) * C;
            scores[0] += tp[0]; scores[1] += tp[1];
            const float *vp = V + ((long)j * 256 + val) * k;
            for (int f = 0; f < k; f++) latent[j * k + f] = vp[f];
        }

        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = latent[j * k + f];
                sum += v; sum_sq += v * v;
            }
            interaction += 0.5f * (sum * sum - sum_sq);
        }
        scores[0] += interaction; scores[1] += interaction;

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(latent);
    return 0;
}
```