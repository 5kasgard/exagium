```hypothesis
We propose a Factorization Machine (FM) of degree 2 with a small embedding dimension (K=4). FMs capture pairwise feature interactions efficiently, which is crucial for tabular data like Adult where interactions (e.g., education × occupation) are known to be predictive. The model has far fewer parameters than an MLP and inference cost is O(d²·K) = 14·13/2·4 = 364 multiply‑adds per example, much cheaper than label‑mlp16. If it does not beat label‑linear (0.8524) on dev, it would show that pairwise interactions alone are insufficient or that K=4 is too small.
```

```python
"""Factorization Machine (degree 2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x464D3231 ("FM21"), d, K, C; float32 w0, W[d*256], V[d*256*K]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    # Parameters
    w0 = np.float32(0.0)
    W = np.zeros((d, 256), dtype=np.float32)          # linear weights
    V = (rng.normal(0, 1.0 / math.sqrt(K), (d, 256, K)).astype(np.float32))  # embeddings

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)

            # Gather embeddings and linear terms
            emb = np.empty((B, d, K), dtype=np.float32)
            lin = np.zeros(B, dtype=np.float32)
            for j in range(d):
                vals = xb[:, j]
                lin += W[j, vals]
                emb[:, j, :] = V[j, vals, :]

            # Interaction: 0.5 * (|sum emb|^2 - sum |emb|^2)
            sum_emb = emb.sum(axis=1)                 # (B, K)
            sum_sq = (emb * emb).sum(axis=1)          # (B, K)
            inter = 0.5 * ((sum_emb * sum_emb).sum(axis=1) - sum_sq.sum(axis=1))

            score = w0 + lin + inter
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb).astype(np.float32)           # (B,)

            # Gradients
            grad_w0 = g.sum()
            grad_W = np.zeros_like(W)
            grad_V = np.zeros_like(V)

            for i in range(B):
                gi = g[i]
                xi = xb[i]
                embi = emb[i]                         # (d, K)
                total_emb = sum_emb[i]                # (K,)
                for j in range(d):
                    v = xi[j]
                    grad_W[j, v] += gi
                    other = total_emb - embi[j]
                    grad_V[j, v] += gi * other

            # L2 regularization
            grad_W += L2 * W
            grad_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * grad_w0
            v_w0 = 0.999 * v_w0 + 0.001 * grad_w0 * grad_w0
            m_hat = m_w0 / (1 - 0.9 ** t)
            v_hat = v_w0 / (1 - 0.999 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # W
            m_W = 0.9 * m_W + 0.1 * grad_W
            v_W = 0.999 * v_W + 0.001 * grad_W * grad_W
            m_hat = m_W / (1 - 0.9 ** t)
            v_hat = v_W / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # V
            m_V = 0.9 * m_V + 0.1 * grad_V
            v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
            m_hat = m_V / (1 - 0.9 ** t)
            v_hat = v_V / (1 - 0.999 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x464D3231  # "FM21"
    header = struct.pack("<4i", magic, d, K, C)
    body = (
        np.array([w0], dtype=np.float32).tobytes() +
        W.astype("<f4").tobytes() +
        V.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per field). Output: one byte class (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;                     // [d][256]
    const float *V = W + (long)d * 256;         // [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Load embeddings for this example
        float emb[14][8]; // max d=14, max K=8 (we use K=4)
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) emb[j][kk] = v[kk];
        }

        float score = w0;
        // Linear part
        for (int j = 0; j < d; j++) score += W[j * 256 + xi[j]];
        // Pairwise interactions
        for (int j1 = 0; j1 < d - 1; j1++) {
            for (int j2 = j1 + 1; j2 < d; j2++) {
                float dot = 0.0f;
                for (int kk = 0; kk < K; kk++) dot += emb[j1][kk] * emb[j2][kk];
                score += dot;
            }
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```