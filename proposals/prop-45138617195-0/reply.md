```hypothesis
Factorization machines (FM) capture pairwise feature interactions with O(k·d) inference cost.
On Adult, interactions like education×occupation are known to improve over linear models.
With latent dimension k=4, the model adds ~36k parameters (still tiny) and ~112 FMA/item.
If dev accuracy ≤ label-linear (0.8524) or instructions/item exceed label-linear significantly, the hypothesis fails.
```

```python
"""Factorization Machine family for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the d byte columns (0-255) gets a linear weight and a k-dim latent vector per class.
Loss: cross-entropy, optimizer: Adam (lr=3e-3, L2=1e-4), batch=64, epochs=min(100, ceil(600000/n)).
Weights layout: magic 0x464D3132, d, k, C; float32 w0[C], L[C][d][256], V[C][d][256][k] (little-endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4  # latent dimension
MAGIC = 0x464D3132  # "FM12"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    w0 = np.zeros(C, dtype=np.float32)
    L = np.zeros((C, d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_L, v_L = np.zeros_like(L), np.zeros_like(L)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward pass
            scores = np.zeros((B, C), dtype=np.float32)
            sum_v = np.zeros((C, B, K), dtype=np.float32)
            sum_v2 = np.zeros((C, B, K), dtype=np.float32)
            for c in range(C):
                scores[:, c] = w0[c]
                for j in range(d):
                    scores[:, c] += L[c, j, xb[:, j]]
                    v = V[c, j, xb[:, j], :]   # (B, K)
                    sum_v[c] += v
                    sum_v2[c] += v * v
                inter = 0.5 * np.sum(sum_v[c]**2 - sum_v2[c], axis=1)
                scores[:, c] += inter

            # Softmax + gradient
            max_s = np.max(scores, axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            p = exp_s / np.sum(exp_s, axis=1, keepdims=True)   # (B, C)
            g = p
            g[np.arange(B), yb] -= 1.0
            g /= B

            # Gradients
            grad_w0 = np.mean(g, axis=0)
            grad_L = np.zeros_like(L)
            grad_V = np.zeros_like(V)

            for c in range(C):
                gc = g[:, c]
                for j in range(d):
                    grad_L[c, j] = np.bincount(xb[:, j], weights=gc, minlength=256)
                for j in range(d):
                    v = V[c, j, xb[:, j], :]
                    diff = sum_v[c] - v
                    for f in range(K):
                        grad_V[c, j, :, f] = np.bincount(xb[:, j], weights=gc * diff[:, f], minlength=256)

            # L2 regularization
            grad_w0 += L2 * w0
            grad_L += L2 * L
            grad_V += L2 * V

            # Adam update
            t += 1
            for param, grad, m, v in [
                (w0, grad_w0, m_w0, v_w0),
                (L, grad_L, m_L, v_L),
                (V, grad_V, m_V, v_V)
            ]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    header = struct.pack("<4i", MAGIC, d, K, C)
    body = (
        w0.astype("<f4").tobytes() +
        L.astype("<f4").tobytes() +
        V.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index (argmax).
 * Weights format: magic 0x464D3132, int32 d, K, C; float32 w0[C], L[C][d][256], V[C][d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16 || d < 1 || d > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    long expected = 16 + 4L * C * (1 + (long)d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *L = w0 + C;
    const float *V = L + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[256];  // C <= 256
    float sum_v[16], sum_v2[16];  // K <= 16

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            float sc = w0[c];
            for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                unsigned char val = xi[j];
                sc += L[((long)c * d + j) * 256 + val];
                const float *v_ptr = V + (((long)c * d + j) * 256 + val) * K;
                for (int f = 0; f < K; f++) {
                    float v = v_ptr[f];
                    sum_v[f] += v;
                    sum_v2[f] += v * v;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) inter += sum_v[f] * sum_v[f] - sum_v2[f];
            sc += 0.5f * inter;
            scores[c] = sc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```