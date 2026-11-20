```hypothesis
Family: label-fm (Factorization Machine classifier). 
Why: Character prediction depends on pairwise interactions between context positions (e.g., "th", "he", "in"). A rank-k FM captures all pairwise interactions with O(k) parameters per feature, adding only ~1.5k ops/item over linear. It should beat label-linear (0.3375) on dev while staying well below label-mlp16 cost (10k Ir/item). 
Failure signal: dev accuracy ≤ 0.3375 or inference cost ≥ 10000 Ir/item.
```
```python
"""Factorization Machine classifier for fixed-size categorical byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16 positions × 256 values = 4096 features. Each example has 16 active features.
Rank-k FM (k=16) computes pairwise interactions: h_f = (sum v_i,f)^2 - sum v_i,f^2.
Linear output layer: scores = h @ W + b (65 classes). Softmax cross-entropy, Adam.
Exported weights: magic 0x464D3130, d, k, C, nfeat; float32 V[nfeat,k], W[k,C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D3130  # "FM10" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    nfeat = d * 256

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Feature indices: for each example, 16 indices in [0, nfeat)
    idx = (np.arange(d, dtype=np.int32) * 256 + X).astype(np.int32)  # (n, d)

    rng = np.random.default_rng(SEED)
    # V: (nfeat, K), W: (K, C), b: (C,)
    V = (rng.normal(0, 0.01, (nfeat, K))).astype(np.float32)
    W = (rng.normal(0, 0.01, (K, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [V, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            idb = idx[batch_idx]          # (B, d)
            yb = y[batch_idx]             # (B,)
            B = len(yb)

            # Forward
            # V_gather: (B, d, K)
            Vg = V[idb]                   # advanced indexing
            sum_v = Vg.sum(axis=1)        # (B, K)
            sum_v2 = (Vg * Vg).sum(axis=1)  # (B, K)
            h = sum_v * sum_v - sum_v2    # (B, K)
            scores = h @ W + b            # (B, C)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B       # (B, C)

            # Backward
            grad_h = grad_scores @ W.T    # (B, K)
            grad_W = h.T @ grad_scores    # (K, C)
            grad_b = grad_scores.sum(axis=0)  # (C,)

            # grad_V: for each factor f, each position j
            # dL/dV[idb[:,j], f] += 2 * grad_h[:,f] * (sum_v[:,f] - Vg[:,j,f])
            grad_V = np.zeros_like(V)
            for f in range(K):
                gh = grad_h[:, f]                     # (B,)
                sv = sum_v[:, f]                      # (B,)
                for j in range(d):
                    idx_j = idb[:, j]                 # (B,)
                    v_j = Vg[:, j, f]                 # (B,)
                    g = 2.0 * gh * (sv - v_j)         # (B,)
                    np.add.at(grad_V[:, f], idx_j, g)

            grads = [grad_V + L2 * V, grad_W + L2 * W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += V.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine classifier: ./infer weights.bin in.rec N
 * Input record: d bytes (categorical). Output: 1 byte class index (argmax).
 * Weights format: magic(0x464D3130), d, K, C; float32 V[d*256][K], W[K][C], b[C]. */
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int nfeat = d * 256;
    long expected = 16L + 4L * ((long)nfeat * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, want %ld\n", nw, expected); return 3; }

    const float *V = (const float *)(hd + 4);
    const float *W = V + (long)nfeat * K;
    const float *b = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h[f] = (sum V[idx,f])^2 - sum V[idx,f]^2
        for (int f = 0; f < K; f++) {
            float sum = 0.0f, sum2 = 0.0f;
            for (int j = 0; j < d; j++) {
                int idx = j * 256 + xi[j];
                float v = V[(long)idx * K + f];
                sum += v;
                sum2 += v * v;
            }
            h[f] = sum * sum - sum2;
        }

        // scores = h @ W + b
        for (int c = 0; c < C; c++) {
            float s = b[c];
            for (int f = 0; f < K; f++) s += h[f] * W[(long)f * C + c];
            scores[c] = s;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```