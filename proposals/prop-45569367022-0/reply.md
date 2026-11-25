```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via factorized embeddings in O(d·K) time (d=14 fields, K=8 factors), adding minimal inference cost over label-linear. Training uses the same Adam setup. Failure: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 1000 Ir/item.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Each of the d input bytes (field) has 256 values. Learns per-field per-value:
  linear weight w[field][value][class] and K-factor embedding v[field][value][class][factor].
Score_c = b_c + sum_j w[j][x_j][c] + 0.5 * sum_f ( (sum_j v[j][x_j][c][f])^2 - sum_j v[j][x_j][c][f]^2 ).
Adam lr=3e-3, L2=1e-4 on w, L2=1e-4 on v, minibatch 64, epochs min(100, ceil(600000/n)). Deterministic seed=1."""
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2_W, L2_V, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100

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

    # Parameters: w[d][256][C], v[d][256][C][K], b[C]
    w = np.zeros((d, 256, C), np.float32)
    v = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mv, vv = np.zeros_like(v), np.zeros_like(v)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # (B, d)
            yi = y[idx]          # (B,)
            B = len(yi)

            # Linear scores: (B, C)
            lin = b + np.sum(w[np.arange(d), xi.T].transpose(1, 0, 2), axis=1)

            # FM interaction per class: (B, C)
            # v_idx: (B, d, C, K) -> sum over fields -> (B, C, K)
            v_idx = v[np.arange(d), xi.T].transpose(1, 0, 2, 3)   # (B, d, C, K)
            sum_v = v_idx.sum(axis=1)                              # (B, C, K)
            sum_v2 = (v_idx ** 2).sum(axis=1)                      # (B, C, K)
            fm = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=2))         # (B, C)

            scores = lin + fm                                      # (B, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yi] -= 1.0
            grad = probs / B                                       # (B, C)

            # Gradients
            # Linear: w[d][256][C] += xi^T @ grad
            # For each field j, value v: count how many times (j,v) appears in batch, weighted by grad
            gw = np.zeros_like(w)
            for j in range(d):
                vals = xi[:, j]                    # (B,)
                g = grad                           # (B, C)
                np.add.at(gw[j], vals, g)          # (256, C)
            gw += L2_W * w
            gb = grad.sum(axis=0) + L2_W * b

            # FM: v[d][256][C][K]
            # d/dv_{j,v,c,f} = sum_{i: x_{i,j}=v} grad_{i,c} * (sum_v_{i,c,f} - v_{j,v,c,f})
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xi[:, j]                    # (B,)
                # sum_v for this batch: (B, C, K)
                sv = sum_v
                sv_minus_v = sv[:, None, :, :] - v_idx   # (B, d, C, K)
                g = grad[:, None, :, None]               # (B, 1, C, 1)
                contrib = g * sv_minus_v                 # (B, d, C, K)
                np.add.at(gv[j], vals, contrib)          # (256, C, K)
            gv += L2_V * v

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for param, grad, m, v in [(w, gw, mw, vw), (v, gv, mv, vv), (b, gb, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights: magic, d, K, C; then w[d][256][C], v[d][256][C][K], b[C] as float32 LE
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class index (argmax).
 * Weights: magic(0x464D3130), d, K, C; float32 w[d][256][C], v[d][256][C][K], b[C]. */
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
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    size_t w_sz = (size_t)d * 256 * C;
    size_t v_sz = (size_t)d * 256 * C * K;
    size_t b_sz = C;
    long expected = 16 + 4L * (w_sz + v_sz + b_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *v = w + w_sz;
    const float *b = v + v_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * C * K);
    float *sum_v2 = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear + bias
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *w_ptr = w + ((size_t)j * 256 + val) * C;
            for (int c = 0; c < C; c++) scores[c] += w_ptr[c];
        }

        // FM interaction per class
        // Zero accumulators
        for (int c = 0; c < C; c++)
            for (int f = 0; f < K; f++) {
                sum_v[c * K + f] = 0.0f;
                sum_v2[c * K + f] = 0.0f;
            }
        // Accumulate v and v^2
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *v_ptr = v + ((size_t)j * 256 * C * K) + (size_t)val * C * K;
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < K; f++) {
                    float vv = v_ptr[c * K + f];
                    sum_v[c * K + f] += vv;
                    sum_v2[c * K + f] += vv * vv;
                }
            }
        }
        // Add 0.5 * (sum_v^2 - sum_v2) to scores
        for (int c = 0; c < C; c++) {
            float fm = 0.0f;
            for (int f = 0; f < K; f++) {
                float sv = sum_v[c * K + f];
                fm += sv * sv - sum_v2[c * K + f];
            }
            scores[c] += 0.5f * fm;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_v); free(sum_v2);
    return 0;
}
```