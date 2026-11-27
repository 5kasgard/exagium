```hypothesis
Family: label-fm (Factorization Machine with k=4 latent dimensions).
Why: Adult income has strong pairwise feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via factorized latent vectors with only O(d*k) extra parameters and compute per sample. With k=4, it adds ~140KB weights and ~250 FLOPs/item (vs ~30 for linear), targeting >0.855 accuracy at <2× linear's instruction cost.
Failure: Dev accuracy ≤ 0.8524 (linear) or Ir/item > 1500 (2× linear).
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Encodes each of d=14 bytes as categorical (256 values).
Learns per-class linear weights w[C][d][256] and shared latent vectors v[C][d][256][k] (k=4).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: int32 magic 0x464D3130, d, k, C; float32 bias[C], w[C][d][256], v[C][d][256][k] (all little-endian)."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D3130  # 'FM10'

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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias[C], w[C,d,256], v[C,d,256,K]
    bias = np.zeros(C, np.float32)
    w = np.zeros((C, d, 256), np.float32)
    v = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    # Adam state
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: gather weights
            # w_gathered: (C, d, B) -> sum over d -> (C, B)
            w_gath = w[:, np.arange(d), xb.T]          # (C, d, B)
            linear = w_gath.sum(axis=1)                # (C, B)

            # v_gathered: (C, d, B, K)
            v_gath = v[:, np.arange(d), xb.T, :]       # (C, d, B, K)
            sum_v = v_gath.sum(axis=1)                 # (C, B, K)
            sum_sq = (v_gath * v_gath).sum(axis=1)     # (C, B, K)

            inter = 0.5 * ((sum_v * sum_v - sum_sq).sum(axis=2))  # (C, B)
            scores = bias[:, None] + linear + inter    # (C, B)

            # Softmax
            max_s = scores.max(axis=0, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=0, keepdims=True)      # (C, B)

            # Gradients
            y_onehot = (yb[None, :] == np.arange(C)[:, None])     # (C, B)
            delta = probs - y_onehot                              # (C, B)

            grad_b = delta.sum(axis=1)                            # (C,)
            grad_w = np.zeros_like(w)
            grad_v = np.zeros_like(v)

            # Vectorized gradient accumulation over features
            for f in range(d):
                vals = xb[:, f]                 # (B,)
                for c in range(C):
                    dc = delta[c]               # (B,)
                    # Linear part
                    np.add.at(grad_w[c, f], vals, dc)
                    # Interaction part
                    sv = sum_v[c]               # (B, K)
                    for l in range(K):
                        term = dc * (sv[:, l] - v[c, f, vals, l])
                        np.add.at(grad_v[c, f, :, l], vals, term)

            # L2 regularization
            grad_w += L2 * w
            grad_v += L2 * v

            # Adam update
            t += 1
            # bias
            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            bias -= LR * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            w -= LR * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
            # v
            m_v = 0.9 * m_v + 0.1 * grad_v
            v_v = 0.999 * v_v + 0.001 * grad_v * grad_v
            v -= LR * (m_v / (1 - 0.9 ** t)) / (np.sqrt(v_v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, d, K, C; bias[C]; w[C,d,256]; v[C,d,256,K]
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weights format: int32 magic(0x464D3130), d, K, C; float32 bias[C], w[C][d][256], v[C][d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims: d=%d K=%d C=%d\n", d, K, C); return 3; }

    size_t expected = 16
        + (size_t)C * 4
        + (size_t)C * d * 256 * 4
        + (size_t)C * d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + C;
    const float *v = w + (size_t)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[4], sum_sq[4], score[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            score[c] = bias[c];
            for (int l = 0; l < K; l++) { sum_v[l] = 0.0f; sum_sq[l] = 0.0f; }
            for (int f = 0; f < d; f++) {
                unsigned char val = xi[f];
                const float *w_ptr = w + ((size_t)c * d * 256 + f * 256 + val);
                score[c] += *w_ptr;
                const float *v_ptr = v + ((size_t)c * d * 256 * K + f * 256 * K + val * K);
                for (int l = 0; l < K; l++) {
                    float vv = v_ptr[l];
                    sum_v[l] += vv;
                    sum_sq[l] += vv * vv;
                }
            }
            for (int l = 0; l < K; l++) {
                score[c] += 0.5f * (sum_v[l] * sum_v[l] - sum_sq[l]);
            }
        }

        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```