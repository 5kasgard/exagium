```hypothesis
Factorization Machines capture pairwise interactions between context positions via low-rank embeddings, crucial for character-level language modeling. With embedding dimension 8, the model adds only ~33k parameters over the linear baseline, enabling it to model patterns like "th"→"e" without overfitting on 24k examples. If dev accuracy does not exceed label-linear (0.3375), the interaction terms are not learning useful patterns.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Model: linear term + FM pairwise interactions (k=8) with class-specific projection.
Weights: magic 0x464D4C31, d, k, C; float32 W_lin[d][256][C], V[d][256][k], U[C][k], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8  # embedding dimension

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
    # Parameters (float32)
    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    U = rng.normal(0, 0.01, (C, K)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [W_lin, V, U, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            lin = np.zeros((bs, C), np.float32)
            emb_sum = np.zeros((bs, K), np.float32)
            emb_sq_sum = np.zeros((bs, K), np.float32)
            for i in range(d):
                xi = Xb[:, i]
                lin += W_lin[i, xi]
                v_i = V[i, xi]
                emb_sum += v_i
                emb_sq_sum += v_i * v_i
            inter = 0.5 * (emb_sum * emb_sum - emb_sq_sum)  # (bs, K)
            logits = lin + inter @ U.T + b  # (bs, C)

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)  # (bs, C)

            # Backward
            grad_b = grad_logits.sum(axis=0)
            grad_U = grad_logits.T @ inter
            grad_inter = grad_logits @ U  # (bs, K)
            grad_emb_sum = grad_inter * emb_sum
            grad_emb_sq_sum = -grad_inter * 0.5
            # grad_V and grad_W_lin
            grad_V = np.zeros_like(V)
            grad_W_lin = np.zeros_like(W_lin)
            for i in range(d):
                xi = Xb[:, i]
                # W_lin gradient
                for c in range(C):
                    grad_W_lin[i, xi, c] += grad_logits[:, c]
                # V gradient
                g_v = grad_emb_sum + grad_emb_sq_sum * 2 * V[i, xi]
                for k in range(K):
                    grad_V[i, xi, k] += g_v[:, k]

            # L2 regularization on all weights except bias
            grad_W_lin += L2 * W_lin
            grad_V += L2 * V
            grad_U += L2 * U

            grads = [grad_W_lin, grad_V, grad_U, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + (long)C * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *V = p; p += (long)d * 256 * K;
    const float *U = p; p += (long)C * K;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        memset(emb_sum, 0, K * sizeof(float));
        memset(emb_sq_sum, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *w = W_lin + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
            const float *ve = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = ve[kk];
                emb_sum[kk] += val;
                emb_sq_sum[kk] += val * val;
            }
        }

        for (int kk = 0; kk < K; kk++)
            inter[kk] = 0.5f * (emb_sum[kk] * emb_sum[kk] - emb_sq_sum[kk]);

        for (int c = 0; c < C; c++) {
            const float *uc = U + (long)c * K;
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) sum += uc[kk] * inter[kk];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```