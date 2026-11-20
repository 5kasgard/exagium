```hypothesis
Factorization Machine (FM) with rank-4 embeddings per class models pairwise feature interactions
efficiently, which are known to boost tabular accuracy over linear models. Adult has many categorical
columns whose interactions (e.g., education × occupation) are predictive. FM adds only ~140 KB
weights and ~2× linear inference cost. If dev accuracy ≤ label-linear (0.8524) or instructions/item
> 1500, the family does not justify its complexity.
```
```python
import json, math, struct, sys

import numpy as np

K = 4
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

    # Build encodings exactly like label-linear
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Concatenate to design matrix Z (n x total_features)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, total_feat = Z.shape

    # Offsets per original column for later export
    offsets = []
    off = 0
    for E in enc:
        offsets.append(off)
        off += E.shape[1]

    # Initialize parameters per class
    w = (rng.uniform(-1, 1, (C, total_feat)) * math.sqrt(6.0 / total_feat)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, total_feat, K)) * math.sqrt(6.0 / (total_feat * K))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [w, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = Z[idx], y[idx]
            B = len(yb)

            # Forward
            linear = Xb @ w.T + b  # (B, C)
            # Interaction: 0.5 * sum_k ( (X v_k)^2 - X (v_k^2) )
            inter = np.zeros((B, C), np.float32)
            for c in range(C):
                Xv = Xb @ V[c]                 # (B, K)
                Xv2 = Xb @ (V[c] * V[c])       # (B, K)
                inter[:, c] = 0.5 * np.sum(Xv * Xv - Xv2, axis=1)
            scores = linear + inter

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            gw = Xb.T @ grad_scores + L2 * w                 # (total_feat, C) -> transpose later
            gb = grad_scores.sum(axis=0)                     # (C,)
            gV = np.zeros_like(V)                            # (C, total_feat, K)
            for c in range(C):
                g = grad_scores[:, c:c+1]                    # (B, 1)
                Xv = Xb @ V[c]                               # (B, K)
                # d/dV = X^T ( g * (Xv - V) )  broadcasting over features
                # (total_feat, B) @ (B, K) -> (total_feat, K)
                gV[c] = Xb.T @ (g * (Xv - V[c])) + L2 * V[c]

            grads = [gw.T, gV, gb]   # match param order [w, V, b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # Export per original column per byte value: linear[d][256][C], embed[d][256][C][K], bias[C]
    d = X.shape[1]
    linear_table = np.zeros((d, 256, C), dtype=np.float32)
    embed_table = np.zeros((d, 256, C, K), dtype=np.float32)
    for j in range(d):
        off = offsets[j]
        E = enc[j]                       # (256, feat_j)
        for c in range(C):
            linear_table[j, :, c] = E @ w[c, off:off+E.shape[1]]
            for k in range(K):
                embed_table[j, :, c, k] = E @ V[c, off:off+E.shape[1], k]

    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += linear_table.astype("<f4").tobytes()
    out += embed_table.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * d * 256 * C + 4L * d * 256 * C * K + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *emb = lin + (long)d * 256 * C;
    const float *bias = emb + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_j = lin + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_j[c];
            const float *emb_j = emb + ((long)j * 256 + v) * C * K;
            for (int c = 0; c < C; c++) {
                for (int kk = 0; kk < K; kk++) {
                    float e = emb_j[(long)c * K + kk];
                    sum_v[kk] += e;
                    sum_sq[kk] += e * e;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) inter += 0.5f * (sum_v[kk] * sum_v[kk] - sum_sq[kk]);
            scores[c] += inter;
            sum_v[kk] = sum_sq[kk] = 0.0f;  // reset for next class
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```