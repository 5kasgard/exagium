```hypothesis
Family: label-fm (Factorization Machine with shared interaction bases).
Why: Optdigits pixels have strong pairwise interactions (strokes, loops) that linear models miss. A rank-k FM captures all pairwise interactions in O(d*k) time (d=64, k=4), far cheaper than an MLP (O(d*H)) while adding expressive power. Shared interaction bases across classes keep parameters small (~0.9 MB) and inference fast.
Failure signal: If dev accuracy < 0.965 (beating label-linear's 0.9613) or instruction count > 8000 Ir/item (label-linear is 5058).
```

```python
"""label-fm: Factorization Machine with shared interaction bases for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights: int32 magic 0x464D3130, d, C, k; float32 w[d][256][C], v[d][256][k], u[C][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 4  # FM rank


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
    k = K
    rng = np.random.default_rng(SEED)

    # Xavier init
    scale_w = math.sqrt(6.0 / (256 + C))
    scale_v = math.sqrt(6.0 / (256 + k))
    scale_u = math.sqrt(6.0 / (C + k))
    w = rng.normal(0, scale_w, (d, 256, C)).astype(np.float32)
    v = rng.normal(0, scale_v, (d, 256, k)).astype(np.float32)
    u = rng.normal(0, scale_u, (C, k)).astype(np.float32)

    params = [w, v, u]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            linear = np.zeros((bsz, C), np.float32)
            S = np.zeros((bsz, k), np.float32)
            Q = np.zeros((bsz, k), np.float32)
            for j in range(d):
                vals = xb[:, j]
                linear += w[j, vals]               # (bsz, C)
                v_j = v[j, vals]                   # (bsz, k)
                S += v_j
                Q += v_j * v_j
            I = 0.5 * (S * S - Q)                  # (bsz, k)
            scores = linear + I @ u.T              # (bsz, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            dscores = (probs / bsz).astype(np.float32)  # (bsz, C)

            # Backward
            dI = dscores @ u                       # (bsz, k)
            dw = np.zeros_like(w)
            dv = np.zeros_like(v)
            du = I.T @ dscores                     # (k, C) -> transpose later
            for j in range(d):
                vals = xb[:, j]
                # vectorized scatter-add via loop (bsz small)
                for i in range(bsz):
                    v_idx = vals[i]
                    dw[j, v_idx] += dscores[i]
                    dv[j, v_idx] += dI[i] * (S[i] - v[j, v_idx])

            grads = [dw, dv, du.T]

            # Adam update with L2
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr += L2 * p
                m, v_adam = state[i]
                m *= 0.9; m += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, C, k, then w, v, u (all float32 LE)
    out = struct.pack("<4i", 0x464D3130, d, C, k)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    out += u.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax).
 * Weights: magic 0x464D3130, int32 d,C,k; float32 w[d][256][C], v[d][256][k], u[C][k]. */
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
    int d = hd[1], C = hd[2], k = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64 || d != 64) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t w_sz = (size_t)d * 256 * C;
    size_t v_sz = (size_t)d * 256 * k;
    size_t u_sz = (size_t)C * k;
    size_t total_floats = w_sz + v_sz + u_sz;
    if (nw != 16 + 4L * (long)total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *v = w + w_sz;
    const float *u = v + v_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *linear = (float *)malloc(sizeof(float) * C);
    float *S = (float *)malloc(sizeof(float) * k);
    float *Q = (float *)malloc(sizeof(float) * k);
    float *I = (float *)malloc(sizeof(float) * k);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        memset(linear, 0, C * sizeof(float));
        memset(S, 0, k * sizeof(float));
        memset(Q, 0, k * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *w_j = w + ((size_t)j * 256 + val) * C;
            for (int c = 0; c < C; c++) linear[c] += w_j[c];

            const float *v_j = v + ((size_t)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vf = v_j[f];
                S[f] += vf;
                Q[f] += vf * vf;
            }
        }

        for (int f = 0; f < k; f++) I[f] = 0.5f * (S[f] * S[f] - Q[f]);

        for (int c = 0; c < C; c++) {
            float sc = linear[c];
            const float *u_c = u + (size_t)c * k;
            for (int f = 0; f < k; f++) sc += u_c[f] * I[f];
            scores[c] = sc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```