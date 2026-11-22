```hypothesis
Family: label-fm (Factorization Machine with one-hot features, rank-8 latent vectors, single logit).
Why: FMs capture pairwise feature interactions at O(d*k) inference cost (d=14, k=8), which is expressive for tabular data with many categorical columns. The one-hot encoding over 256 values per column matches the input format exactly. With ~32k parameters, it should exceed label-linear's 0.8524 accuracy while staying well below label-mlp16's 2325 Ir/item.
Failure: If dev accuracy ≤ 0.8524 or Ir/item ≥ 719, the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Weights layout (little-endian):
  int32 magic=0x464D4C31, d=14, k=8, C=2
  float32 w0
  float32 w[d*256]
  float32 V[d*256*k]
"""
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100

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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]

            # Forward pass: compute logit for each example
            # logit = w0 + sum_j w[j, x_j] + 0.5 * ( (sum_j V[j, x_j])^2 - sum_j V[j, x_j]^2 )
            sum_w = np.full(b, w0, dtype=np.float32)
            sum_V = np.zeros((b, K), dtype=np.float32)
            sum_V_sq = np.zeros((b, K), dtype=np.float32)
            for j in range(d):
                vals = xb[:, j]
                sum_w += w[j, vals]
                v = V[j, vals]          # (b, K)
                sum_V += v
                sum_V_sq += v * v
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1)
            logit = sum_w + inter

            # Logistic loss gradient: sigmoid(logit) - y
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = p - yb.astype(np.float32)   # (b,)

            # Backward pass
            g_w0 = np.mean(grad)
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                # g_w[j, v] = mean_{i: xb[i,j]=v} grad[i]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g_w[j, v] = np.mean(grad[mask])
                # g_V[j, v] = sum_{i: xb[i,j]=v} grad[i] * (sum_V[i] - V[j, v])
                # We can accumulate per example then aggregate.
                # For each example i, contribution to V[j, xb[i,j]] is grad[i] * (sum_V[i] - V[j, xb[i,j]])
                # We'll loop over batch (small) for simplicity.
                for i in range(b):
                    v = vals[i]
                    g_V[j, v] += grad[i] * (sum_V[i] - V[j, v])

            # Add L2 regularization
            g_w += L2 * w
            g_V += L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_hat_w = m_w / (1 - beta1 ** t)
            v_hat_w = v_w / (1 - beta2 ** t)
            w -= LR * m_hat_w / (np.sqrt(v_hat_w) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

    # Write weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, 2)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1).
 * Weights format: int32 magic=0x464D4C31, d, k, C; float32 w0, w[d*256], V[d*256*k] */
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
    if (d != 14 || K != 8 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_V_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        for (int u = 0; u < K; u++) { sum_V[u] = 0.0f; sum_V_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = Vjv[u];
                sum_V[u] += val;
                sum_V_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_V[u] * sum_V[u] - sum_V_sq[u];
        logit += 0.5f * inter;

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```