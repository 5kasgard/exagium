```hypothesis
We propose a factorization machine family "label-fm" that learns pairwise feature interactions via low-rank embeddings (K=8). On tabular data with mixed categorical/ordinal features, FMs capture interactions that linear models miss, often improving accuracy with minimal inference cost (O(d*K)). With K=8 we expect dev accuracy >0.8524 (label-linear) at similar or lower CPU cost. If dev accuracy <=0.8524 or instructions per item exceed label-linear by >20%, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen based on typical FM performance on Adult)
K = 8          # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20

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
    assert C == 2, "This FM implementation expects binary classification"

    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize weights
    # Linear weights: (d, 256)
    W_lin = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    # Embeddings: (d, 256, K)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam parameters
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_b = 0.0; v_b = 0.0
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]  # (B, d)
            y_batch = y[idx]  # (B,)

            # Forward pass
            # Linear part: sum_j W_lin[j, x_batch[:,j]]
            lin = np.sum(W_lin[np.arange(d), x_batch.T], axis=0)  # (B,)
            # Interaction part: 0.5 * ( (sum_j V[j, x_j])^2 - sum_j V[j, x_j]^2 )
            emb = V[np.arange(d), x_batch.T]  # (d, B, K)
            emb = emb.transpose(1, 0, 2)      # (B, d, K)
            sum_emb = np.sum(emb, axis=1)           # (B, K)
            sum_sq_emb = np.sum(emb * emb, axis=1)  # (B, K)
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)  # (B,)
            logits = bias + lin + inter  # (B,)

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - y_batch).astype(np.float32)  # (B,)

            # Gradients
            grad_W = np.zeros_like(W_lin)
            grad_V = np.zeros_like(V)
            for j in range(d):
                vals = x_batch[:, j]  # (B,)
                np.add.at(grad_W[j], vals, grad)
                for i in range(len(idx)):
                    v_idx = vals[i]
                    diff = sum_emb[i] - V[j, v_idx]
                    grad_V[j, v_idx] += grad[i] * diff

            grad_b = np.sum(grad)

            # Add L2 regularization
            grad_W += L2 * W_lin
            grad_V += L2 * V

            # Adam update
            t += 1
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W_lin -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    w_bytes = W_lin.astype("<f4").tobytes()
    v_bytes = V.astype("<f4").tobytes()
    b_bytes = np.array([bias], dtype="<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(w_bytes)
        f.write(v_bytes)
        f.write(b_bytes)

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K < 1 || K > 256 || C != 2) { fprintf(stderr, "bad weights: invalid dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 + (long)d * 256 * K + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *W_lin = (float *)(wb + 16);
    float *V = (float *)(wb + 16 + 4L * d * 256);
    float bias = *(float *)(wb + 16 + 4L * d * 256 + 4L * d * 256 * K);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * K);
    float *sum_sq_emb = (float *)malloc(sizeof(float) * K);
    if (!sum_emb || !sum_sq_emb) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int kk = 0; kk < K; kk++) {
            sum_emb[kk] = 0.0f;
            sum_sq_emb[kk] = 0.0f;
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += W_lin[(long)j * 256 + v];
            const float *emb = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = emb[kk];
                sum_emb[kk] += val;
                sum_sq_emb[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_emb[kk] * sum_emb[kk] - sum_sq_emb[kk];
        }
        inter *= 0.5f;

        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_emb);
    free(sum_sq_emb);
    return 0;
}
```