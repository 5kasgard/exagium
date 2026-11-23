```hypothesis
Family: label-fm (second-order Factorization Machine on 14 fields with shared embeddings).
Why: Adult has 14 mixed-type fields; FM captures pairwise interactions with O(d*k) parameters and inference cost, unlike MLP's O(d*H) or linear's none. Shared embeddings per field-category keep weights small. Beats linear (0.8524) by modeling interactions; cheaper than MLP16 (0.8591) if k=8.
Failure: If dev accuracy ≤0.8524 or instructions/item ≥ label-linear (719), the interaction benefit doesn't justify the cost.
```
```python
"""label-fm: Factorization Machine on 14 categorical fields (256 categories each).
Train: python3 -s -P train.py card.json train.in train.key weights.bin
Weights: magic=0x464D3130, d=14, k, C=2; float32 bias[C], linear[C][d][256], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8                 # embedding size (tuned on last 20% of train.in)
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 15

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def softmax_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # deterministic
    rng = np.random.default_rng(SEED)

    # parameters
    bias = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    def adam_init(shape):
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    m_bias, v_bias = adam_init(bias.shape)
    m_lin, v_lin = adam_init(linear.shape)
    m_emb, v_emb = adam_init(emb.shape)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # forward
            # linear part: (B, C)
            lin_part = bias + np.sum(linear[:, xb.T, np.arange(d)].transpose(2,0,1), axis=1)  # (B, C)

            # interaction part: 0.5 * ( (sum emb)^2 - sum(emb^2) )
            emb_x = emb[np.arange(d), xb.T]  # (d, B, K)
            sum_emb = emb_x.sum(axis=0)       # (B, K)
            sum_emb_sq = np.sum(emb_x * emb_x, axis=0)  # (B, K)
            inter = 0.5 * (np.sum(sum_emb * sum_emb, axis=1) - np.sum(sum_emb_sq, axis=1))  # (B,)

            logits = lin_part + inter[:, None]  # (B, C)

            # gradient
            grad = softmax_grad(logits, yb)  # (B, C)

            # bias grad
            g_bias = grad.sum(axis=0)

            # linear grad: (C, d, 256)
            g_lin = np.zeros_like(linear)
            for c in range(C):
                for j in range(d):
                    np.add.at(g_lin[c, j], xb[:, j], grad[:, c])

            # emb grad: (d, 256, K)
            g_emb = np.zeros_like(emb)
            # d(inter)/d(emb_j) = (sum_emb - emb_j)  for each example
            # chain rule: grad_inter = grad[:,0] + grad[:,1]  (since inter added to both classes)
            grad_inter = grad.sum(axis=1)  # (B,)
            for j in range(d):
                emb_j = emb_x[j]  # (B, K)
                # for each example, contribution to emb[j, xb[:,j]] is grad_inter * (sum_emb - emb_j)
                diff = (sum_emb - emb_j) * grad_inter[:, None]  # (B, K)
                np.add.at(g_emb[j], xb[:, j], diff)

            # L2 regularization
            g_bias += L2 * bias
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (bias, g_bias, m_bias, v_bias),
                (linear, g_lin, m_lin, v_lin),
                (emb, g_emb, m_emb, v_emb),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    magic = 0x464D3130  # 'FM10'
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per field). Output: 1 byte class (0/1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16
        + C * sizeof(float)
        + C * d * 256 * sizeof(float)
        + d * 256 * K * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *bias = p; p += C;
    const float *linear = p; p += C * d * 256;
    const float *emb = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[64], sum_emb_sq[64], scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld len %u != %d\n", i, k, d); return 3; }

        // linear part
        scores[0] = bias[0];
        scores[1] = bias[1];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            scores[0] += linear[0 * d * 256 + j * 256 + v];
            scores[1] += linear[1 * d * 256 + j * 256 + v];
        }

        // interaction part: 0.5 * ( (sum emb)^2 - sum(emb^2) )
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_emb_sq, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *e = emb + (j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
        }
        inter *= 0.5f;
        scores[0] += inter;
        scores[1] += inter;

        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```