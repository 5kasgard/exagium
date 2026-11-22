```hypothesis
Family: label-fm (Factorization Machine with class-specific embeddings, K=4).
Why: Adult is a tabular dataset where pairwise feature interactions (e.g., education × occupation) are known to boost accuracy over linear models. A factorization machine captures all pairwise interactions in O(d·K) time using the identity ½[(∑v_i)² - ∑v_i²]. With d=14, C=2, K=4 the model has ~36k parameters (≈143 KiB), trains in seconds, and runs at ~2× the cost of label-linear but should exceed its 0.8524 dev accuracy.
Failure signal: dev accuracy ≤ 0.8524 or instructions/item > 1500 (2× label-linear).
```

```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic=0x464D5F31, d, K, C (int32)
  linear_w[d][256][C] (float32)
  linear_b[C] (float32)
  emb[d][256][C][K] (float32)"""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    lin_w = np.zeros((d, 256, C), np.float32)
    lin_b = np.zeros(C, np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)

    params = [lin_w, lin_b, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # ---- forward ----
            # linear part
            lin_out = lin_b.copy()
            for j in range(d):
                lin_out += lin_w[j, xb[:, j]]          # (B, C)

            # FM interaction part per class
            fm_out = np.zeros((B, C), np.float32)
            for c in range(C):
                sum_v = np.zeros((B, K), np.float32)
                sum_v2 = np.zeros((B, K), np.float32)
                for j in range(d):
                    v = emb[j, xb[:, j], c]            # (B, K)
                    sum_v += v
                    sum_v2 += v * v
                fm_out[:, c] = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)

            scores = lin_out + fm_out                  # (B, C)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B                    # (B, C)

            # ---- backward ----
            # linear bias
            grad_lin_b = grad_scores.sum(axis=0)

            # linear weights
            grad_lin_w = np.zeros_like(lin_w)
            for j in range(d):
                for c in range(C):
                    np.add.at(grad_lin_w[j, :, c], xb[:, j], grad_scores[:, c])

            # embeddings
            grad_emb = np.zeros_like(emb)
            for c in range(C):
                g = grad_scores[:, c]                  # (B,)
                sum_v = np.zeros((B, K), np.float32)
                for j in range(d):
                    v = emb[j, xb[:, j], c]
                    sum_v += v
                for j in range(d):
                    v = emb[j, xb[:, j], c]
                    # d/dv [ ½( (∑v)² - ∑v² ) ] = (∑v) - v
                    gv = (sum_v - v) * g[:, None]      # (B, K)
                    np.add.at(grad_emb[j, :, c], xb[:, j], gv)

            grads = [grad_lin_w, grad_lin_b, grad_emb]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * (m_hat / (np.sqrt(v_hat) + 1e-8) + L2 * p)

    # ---- write weights ----
    magic = 0x464D5F31  # "FM_1"
    out = struct.pack("<4i", magic, d, K, C)
    out += lin_w.astype("<f4").tobytes()
    out += lin_b.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte = argmax class. */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad header values\n"); return 3; }

    size_t lin_w_sz = (size_t)d * 256 * C;
    size_t lin_b_sz = C;
    size_t emb_sz = (size_t)d * 256 * C * K;
    size_t total_floats = lin_w_sz + lin_b_sz + emb_sz;
    if (nw != 16 + 4L * total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin_w = (const float *)(hd + 4);
    const float *lin_b = lin_w + lin_w_sz;
    const float *emb = lin_b + lin_b_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = lin_b[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) scores[c] += lin_w[((size_t)j * 256 + v) * C + c];
        }

        for (int c = 0; c < C; c++) {
            memset(sum_v, 0, K * sizeof(float));
            memset(sum_v2, 0, K * sizeof(float));
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                const float *e = emb + (((size_t)j * 256 + v) * C + c) * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = e[kk];
                    sum_v[kk] += val;
                    sum_v2[kk] += val * val;
                }
            }
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```