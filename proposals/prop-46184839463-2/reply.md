```hypothesis
Family: label-fm (Factorization Machine with low-rank pairwise interactions).
Why: Character-level language modeling benefits from modeling pairwise interactions between context bytes (e.g., bigrams). A linear model cannot capture these; a full MLP is heavier. FM adds a low-rank interaction layer (embedding dim K=16) on top of a linear baseline, adding only ~65k parameters and ~2k multiply-adds per item. This should beat label-linear (0.3375) while staying far cheaper than label-gbdt.
Failure signal: If dev accuracy ≤ 0.3375 or instructions/item > 30k, the interaction layer does not justify its cost.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
K = 16          # embedding dimension for pairwise interactions
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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

    num_features = d * 256
    # Linear weights
    W1 = rng.normal(0, 0.01, (num_features, C)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    # Embeddings for FM
    V = rng.normal(0, 0.01, (num_features, K)).astype(np.float32)
    # Interaction projection
    W2 = rng.normal(0, 0.01, (K, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, V, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute feature indices for each example: (n, d)
    feat_idx = X + np.arange(d) * 256  # shape (n, d)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = feat_idx[idx]  # (B, d)
            yy = y[idx]

            # Linear part
            linear = W1[batch_idx].sum(axis=1) + b1  # (B, C)

            # FM part
            v = V[batch_idx]  # (B, d, K)
            sum_v = v.sum(axis=1)  # (B, K)
            sum_sq = (v * v).sum(axis=1)  # (B, K)
            h = 0.5 * (sum_v * sum_v - sum_sq)  # (B, K)
            interaction = h @ W2 + b2  # (B, C)

            scores = linear + interaction  # (B, C)

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(len(yy)), yy] -= 1.0
            grad_scores /= len(yy)  # (B, C)

            # Gradients for linear
            grad_W1 = np.zeros_like(W1)
            np.add.at(grad_W1, batch_idx.reshape(-1), grad_scores.repeat(d, axis=0).reshape(-1, C))
            grad_b1 = grad_scores.sum(axis=0)

            # Gradients for FM
            grad_h = grad_scores @ W2.T  # (B, K)
            # h = 0.5*(sum_v^2 - sum_sq) -> dh/dv_p = sum_v - v_p
            grad_v = grad_h[:, None, :] * (sum_v[:, None, :] - v)  # (B, d, K)
            grad_V = np.zeros_like(V)
            np.add.at(grad_V, batch_idx.reshape(-1), grad_v.reshape(-1, K))
            grad_W2 = h.T @ grad_scores  # (K, C)
            grad_b2 = grad_scores.sum(axis=0)

            # L2 regularization (not on biases)
            grad_W1 += L2 * W1
            grad_V += L2 * V
            grad_W2 += L2 * W2

            grads = [grad_W1, grad_b1, grad_V, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic, d, K, C (int32 little-endian)
    # Then: W1 (d*256*C float32), b1 (C), V (d*256*K), W2 (K*C), b2 (C)
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (linear + low-rank pairwise interactions) for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte: argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16
        + 4L * d * 256 * C       /* W1 */
        + 4L * C                 /* b1 */
        + 4L * d * 256 * K       /* V  */
        + 4L * K * C             /* W2 */
        + 4L * C;                /* b2 */
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *W1 = (const float *)(hd + 4);
    const float *b1 = W1 + (long)d * 256 * C;
    const float *V  = b1 + C;
    const float *W2 = V  + (long)d * 256 * K;
    const float *b2 = W2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *h = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Linear part */
        for (int c = 0; c < C; c++) lin[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *w = W1 + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin[c] += w[c];
        }

        /* FM part */
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_sq[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float val = v[u];
                sum_v[u] += val;
                sum_sq[u] += val * val;
            }
        }
        for (int u = 0; u < K; u++) {
            h[u] = 0.5f * (sum_v[u] * sum_v[u] - sum_sq[u]);
        }
        for (int c = 0; c < C; c++) inter[c] = b2[c];
        for (int u = 0; u < K; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) inter[c] += hu * w[c];
        }

        /* Combine and argmax */
        unsigned char best = 0;
        float best_score = lin[0] + inter[0];
        for (int c = 1; c < C; c++) {
            float sc = lin[c] + inter[c];
            if (sc > best_score) { best_score = sc; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```