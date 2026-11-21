```hypothesis
Family: label-fm (factorization machine with learned embeddings per feature value).
Why: Adult has 14 mixed-type features with many categorical values; FM captures pairwise interactions
via low-rank embeddings (O(d*k) inference) and often beats linear models on sparse tabular data.
Failure: If dev accuracy <= 0.8524 (label-linear) or Ir/item >= 1000.
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Learns per-feature-value embeddings (dim K) and linear weights.
Exports: magic 0x464D3130, d, K, C; float32 W[d][256] (linear), V[d][256][K] (embeddings), b[C]."""
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-5, 256, 800000, 100

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
    d, C = card["input"]["bytes"], card["classes"]
    rng = np.random.default_rng(SEED)

    # Init: linear weights per feature value, embeddings per feature value
    W = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    b = np.zeros(C, np.float32)

    n = X.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            # Forward: scores = b + sum_j W[j, x_j] + 0.5 * sum_k ((sum_j V[j, x_j, k])^2 - sum_j V[j, x_j, k]^2)
            linear = b + np.sum(W[np.arange(d)[:, None], xb.T, :], axis=0)  # (m, C)
            sum_v = np.sum(V[np.arange(d)[:, None], xb.T, :], axis=0)       # (m, K)
            sum_v2 = np.sum(V[np.arange(d)[:, None], xb.T, :] ** 2, axis=0) # (m, K)
            inter = 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1, keepdims=True) # (m, 1)
            scores = linear + inter

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            probs /= m  # (m, C)

            # Gradients
            # dL/dW[j, v, c] = sum_{i: x_ij=v} probs[i, c] + L2 * W[j, v, c]
            # dL/dV[j, v, k] = sum_{i: x_ij=v} (sum_v[i, k] - V[j, v, k]) * probs[i, :].sum()? Wait.
            # Actually inter = 0.5 * sum_k ( (sum_j V_jk)^2 - sum_j V_jk^2 )
            # d(inter)/dV[j, v, k] = (sum_v[k] - V[j, v, k]) * I(x_j=v)
            # And this inter is added to ALL classes equally? No, in standard FM for classification,
            # the interaction term is per-class? Or shared? Let's use shared interaction (binary) or per-class.
            # For multi-class, we can have per-class interaction or shared. Let's do shared for simplicity
            # and because C=2 here. But the contract says classes=2. Let's do per-class embeddings V[j, v, k, c]?
            # That would be too large. Standard FM uses shared V for binary, or V per class for multi-class.
            # Since C=2, we can do shared V and per-class linear. But the gradient above assumed inter is (m,1) added to all classes.
            # That means dL/dV = (sum over c probs[:, c]) * (sum_v - V) = 0 * ... = 0. Bad.
            # So interaction MUST be per-class. Let's do V[d, 256, K, C]? Too big: 14*256*8*2 = 57KB, fine.
            # Actually, let's restructure: scores_c = b_c + sum_j W[j, x_j, c] + 0.5 * sum_k ( (sum_j V[j, x_j, k, c])^2 - sum_j V[j, x_j, k, c]^2 )
            # This is per-class FM. Let's redo.

            pass  # Restart with correct per-class FM

    # The above approach is flawed. Let me write a correct per-class FM implementation.

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Weights: magic 0x464D3130, d, K, C; float32 W[d][256][C], V[d][256][K][C], b[C] */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *V = W + (long)d * 256 * C;
    const float *b = V + (long)d * 256 * K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K * C);
    float *sum_v2 = malloc(sizeof(float) * K * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int kk = 0; kk < K * C; kk++) { sum_v[kk] = 0.0f; sum_v2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *Wjv = W + ((long)j * 256 + v) * C;
            const float *Vjv = V + ((long)j * 256 + v) * K * C;
            for (int c = 0; c < C; c++) scores[c] += Wjv[c];
            for (int kk = 0; kk < K; kk++) {
                for (int c = 0; c < C; c++) {
                    float val = Vjv[kk * C + c];
                    int idx = kk * C + c;
                    sum_v[idx] += val;
                    sum_v2[idx] += val * val;
                }
            }
        }
        for (int kk = 0; kk < K; kk++) {
            for (int c = 0; c < C; c++) {
                int idx = kk * C + c;
                scores[c] += 0.5f * (sum_v[idx] * sum_v[idx] - sum_v2[idx]);
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```