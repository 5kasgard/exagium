```hypothesis
Factorization Machine (FM) with rank-8 latent vectors captures pairwise feature interactions crucial for tabular data like Adult (e.g., education×occupation, capital-gain×hours). Direct embedding lookup on the pre-binned byte features avoids one-hot expansion, keeping weights small (~130 KB). Binary cross-entropy with Adam matches label-linear's training setup. Inference is O(d·k) = 112 multiply-adds per item. Expect dev accuracy ≥ 0.858 (beating linear's 0.8524) at ~500 Ir/item. If accuracy < 0.855, rank-8 is insufficient for this task's interaction complexity.
```

```python
"""Factorization Machine (rank-k) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights: magic 0x464D0001, d, k, C(=2), float32 w0, W_lin[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 50


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: w0 (1), W_lin (d,256), V (d,256,K)
    w0 = np.array([math.log((y==1).sum() / (y==0).sum())], dtype=np.float32) if (y==0).sum() else np.array([0.0], dtype=np.float32)
    W_lin = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    params = [w0, W_lin, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Linear term: w0 + sum_j W_lin[j, xb[:,j]]
            lin = w0 + np.sum(W_lin[np.arange(d), xb.T], axis=0)  # (B,)

            # Interaction: 0.5 * sum_k ( (sum_j V[j,x_j,k])^2 - sum_j V[j,x_j,k]^2 )
            V_sel = V[np.arange(d), xb.T]          # (d, B, K)
            S = np.sum(V_sel, axis=0)              # (B, K)
            Q = np.sum(V_sel * V_sel, axis=0)      # (B, K)
            inter = 0.5 * np.sum(S * S - Q, axis=1)  # (B,)

            score = lin + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - yb).astype(np.float32)  # (B,)

            # Gradients
            gw0 = np.array([grad_score.mean()], dtype=np.float32)
            gW = np.zeros_like(W_lin)
            gV = np.zeros_like(V)

            for j in range(d):
                vals = xb[:, j]
                # linear grad
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        gW[j, v] = grad_score[mask].mean()
                # embedding grad: (S_k - V[j,v,k]) * grad_score
                V_j = V_sel[j]          # (B, K)
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        gV[j, v] = np.mean((S[mask] - V_j[mask]) * grad_score[mask, None], axis=0)

            # L2
            gW += L2 * W_lin
            gV += L2 * V

            grads = [gw0, gW, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, K, C, w0, W_lin flat, V flat
    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (pre-binned features). Output: 1 byte class (0/1).
 * Weights: magic 0x464D0001, d, K, C(=2), float32 w0, W_lin[d*256], V[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int l = 0; l < K; l++) S[l] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += W_lin[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int l = 0; l < K; l++) S[l] += Vjv[l];
        }

        for (int l = 0; l < K; l++) {
            float s = S[l];
            score += 0.5f * s * s;
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                float vjv = V[((long)j * 256 + v) * K + l];
                score -= 0.5f * vjv * vjv;
            }
        }

        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```