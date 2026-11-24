```hypothesis
Factorization Machine (FM) with shared embeddings (k=8) captures pairwise feature interactions crucial for Adult (e.g., education×occupation) while adding only ~140 KB weights and O(d·k) inference. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower CPU cost. Failure if dev accuracy ≤ 0.8524 or Ir/item > 2500.
```
```python
"""Factorization Machine for tabular label tasks: fixed-size byte items, ordinal/categorical per card.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic, seed=1, single thread)
Exports per-byte lookup tables: magic 0x464D3130, d, k, C; float32 bias[C], linear[d][256][C], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Build encoders and compute encoded dimension D
    enc = []
    offsets = []
    off = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256, 1)
            enc.append(E)
            offsets.append((off, off + 1))
            off += 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            offsets.append((off, off + len(vals)))
            off += len(vals)
    D = off

    # Build full encoded matrix Z (n, D)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # FM parameters
    linear = (rng.uniform(-0.01, 0.01, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    emb = (rng.uniform(-0.01, 0.01, (D, K)) * math.sqrt(6.0 / (D + K))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [linear, emb, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            linear_part = z @ linear  # (bsz, C)
            a = z @ emb               # (bsz, K)
            a_sq = (z**2) @ (emb**2)  # (bsz, K)
            inter = 0.5 * (a**2 - a_sq).sum(axis=1, keepdims=True)  # (bsz, 1)
            scores = linear_part + inter + bias  # (bsz, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            d_scores = probs / bsz  # (bsz, C)

            # Gradients
            d_linear = z.T @ d_scores + L2 * linear
            d_bias = d_scores.sum(axis=0)

            d_inter = d_scores.sum(axis=1)  # (bsz,)
            d_a = d_inter[:, None] * a      # (bsz, K)
            term1 = z.T @ d_a               # (D, K)
            term2 = emb * ((z**2).T @ d_inter)[:, None]  # (D, K)
            d_emb = (term1 - term2) / bsz + L2 * emb

            grads = [d_linear, d_emb, d_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-byte lookup tables
    T_linear = np.zeros((d, 256, C), dtype=np.float32)
    T_emb = np.zeros((d, 256, K), dtype=np.float32)
    for j in range(d):
        lo, hi = offsets[j]
        E = enc[j]  # (256, Dj)
        T_linear[j] = E @ linear[lo:hi]
        T_emb[j] = E @ emb[lo:hi]

    # Pack weights
    magic = 0x464D3130  # 'FM10'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += bias.astype("<f4").tobytes()
    out += T_linear.astype("<f4").tobytes()
    out += T_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);
    float score[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        score[0] = bias[0]; score[1] = bias[1];
        for (int f = 0; f < K; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin = linear + ((long)j * 256 + v) * C;
            score[0] += lin[0]; score[1] += lin[1];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            float inter = 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
            score[0] += inter; score[1] += inter;
        }
        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```