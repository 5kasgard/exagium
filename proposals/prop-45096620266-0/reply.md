```hypothesis
label-fm: Factorization Machine with per-class embeddings (K=8). Captures pairwise feature interactions crucial for tabular data with mixed categorical/ordinal features. Adds only O(d*K) inference cost over linear model. If dev accuracy <= label-linear (0.8524) or inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8


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
    rng = np.random.default_rng(SEED)

    # Parameters: L (linear), V (embeddings), b (bias)
    L = np.zeros((C, d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_L = np.zeros_like(L); v_L = np.zeros_like(L)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            scores = np.zeros((B, C), dtype=np.float32)
            sum_emb = np.zeros((C, B, K), dtype=np.float32)   # (C, B, K)
            for c in range(C):
                # Linear part
                lin = np.zeros(B, dtype=np.float32)
                for j in range(d):
                    lin += L[c, j, xb[:, j]]
                # Embeddings sum
                se = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    se += V[c, j, xb[:, j], :]
                sum_emb[c] = se
                # Interaction
                inter = 0.5 * np.sum(se * se - np.sum(V[c, :, xb, :] ** 2, axis=1), axis=1)
                scores[:, c] = lin + inter + b[c]

            # Softmax cross-entropy gradient
            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Backward pass
            grad_L = np.zeros_like(L)
            grad_V = np.zeros_like(V)
            grad_b = np.zeros_like(b)

            for c in range(C):
                g = grad_scores[:, c]           # (B,)
                grad_b[c] = g.sum()
                for j in range(d):
                    vals = xb[:, j]
                    # Linear gradient
                    np.add.at(grad_L[c, j], vals, g)
                    # Interaction gradient
                    v_batch = V[c, j, vals, :]          # (B, K)
                    grad_V_batch = g[:, None] * (sum_emb[c] - v_batch)
                    for u in range(K):
                        np.add.at(grad_V[c, j, vals, u], np.arange(B), grad_V_batch[:, u])

            # L2 regularization
            grad_L += L2 * L
            grad_V += L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_L = beta1 * m_L + (1 - beta1) * grad_L
            v_L = beta2 * v_L + (1 - beta2) * grad_L * grad_L
            m_hat_L = m_L / (1 - beta1 ** t)
            v_hat_L = v_L / (1 - beta2 ** t)
            L -= LR * m_hat_L / (np.sqrt(v_hat_L) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            b -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Export weights
    magic = 0x464D4C31  # "FM L1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    for c in range(C):
        out += L[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
        out += struct.pack("<f", b[c])
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    unsigned char *wdata = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wdata;
    if (hdr[0] != 0x464D4C31) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hdr[1];
    int K = hdr[2];
    int C = hdr[3];
    if (d != 14 || C != 2 || K <= 0 || K > 64) { fprintf(stderr, "bad weights: invalid dimensions\n"); return 3; }
    long expected = 16 + (long)C * ((long)d * 256 + (long)d * 256 * K + 1) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights: size mismatch %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wdata + 16);
    float *L[2];
    float *V[2];
    float b[2];
    for (int c = 0; c < C; c++) {
        L[c] = ptr;
        ptr += d * 256;
        V[c] = ptr;
        ptr += (long)d * 256 * K;
        b[c] = *ptr++;
    }

    unsigned char *xdata = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *sum_emb = (float *)malloc(sizeof(float) * C * K);
    float *sum_emb_sq = (float *)malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(xdata, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float scores[2] = {b[0], b[1]};
        for (int c = 0; c < C; c++) {
            for (int u = 0; u < K; u++) {
                sum_emb[c*K + u] = 0.0f;
                sum_emb_sq[c*K + u] = 0.0f;
            }
        }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += L[c][(long)j * 256 + val];
                const float *v = V[c] + ((long)j * 256 + val) * K;
                for (int u = 0; u < K; u++) {
                    float vu = v[u];
                    sum_emb[c*K + u] += vu;
                    sum_emb_sq[c*K + u] += vu * vu;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int u = 0; u < K; u++) {
                float s = sum_emb[c*K + u];
                inter += s * s - sum_emb_sq[c*K + u];
            }
            scores[c] += 0.5f * inter;
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```