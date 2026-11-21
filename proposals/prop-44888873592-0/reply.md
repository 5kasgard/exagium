```hypothesis
Family: label-fm (Factorization Machine, degree 2, rank 4).
Adult income benefits from feature interactions (e.g., education×occupation, marital-status×sex).
FM models all pairwise interactions via low-rank factors, adding only ~14K parameters over linear.
Inference: 14 linear terms + 91 dot-products of size 4 (~400 FLOPs), faster than MLP.
Failure: if dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 719.
```
```python
"""Factorization Machine (degree 2, rank 4) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D31, d, k, C, w0, w[d][256], V[d][256][k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]

    # Parameters: w0 (scalar), w[d, 256], V[d, 256, K]
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward pass
            # Linear part: w0 + sum_j w[j, x_j]
            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (bsz,)

            # Interaction part: sum_{i<j} <V[i, x_i], V[j, x_j]>
            # Compute sum of V vectors for each example: S = sum_j V[j, x_j]  (bsz, K)
            S = np.sum(V[np.arange(d), xb.T], axis=0)  # (bsz, K)
            # For each example, sum_{i<j} <v_i, v_j> = 0.5 * (|S|^2 - sum_j |v_j|^2)
            S_sq = np.sum(S * S, axis=1)  # (bsz,)
            v_sq = np.sum(V[np.arange(d), xb.T] ** 2, axis=(0, 2))  # (bsz,)
            inter = 0.5 * (S_sq - v_sq)

            score = lin + inter
            p = sigmoid(score)
            grad = (p - yb).astype(np.float32)  # (bsz,)

            # Gradients
            g_w0 = np.mean(grad)

            # Linear weights gradient: for each active (j, v), add grad[example]
            g_w = np.zeros_like(w)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, grad)
            g_w /= bsz
            g_w += L2 * w

            # V gradients: for each active (j, v), grad * (S - V[j, v])
            g_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                # S_minus_vj = S - V[j, vals]  (bsz, K)
                S_minus_vj = S - V[j, vals]
                # Outer product: grad[:, None] * S_minus_vj  -> (bsz, K)
                # Accumulate per (j, val)
                for k in range(K):
                    np.add.at(g_V[j, :, k], vals, grad * S_minus_vj[:, k])
            g_V /= bsz
            g_V += L2 * V

            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Pack weights: magic, d, K, C, w0, w[d][256], V[d][256][K]
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2, rank K) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (0 or 1). Weights: see train.py. */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad header\n"); return 3; }

    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += (size_t)d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        float score = w0;
        for (int j = 0; j < d; j++) score += w[(size_t)j * 256 + xi[j]];

        // Interaction part: 0.5 * (|sum V|^2 - sum |V|^2)
        for (int kk = 0; kk < K; kk++) S[kk] = 0.0f;
        float sum_v_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((size_t)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = vj[kk];
                S[kk] += v;
                sum_v_sq += v * v;
            }
        }
        float S_sq = 0.0f;
        for (int kk = 0; kk < K; kk++) S_sq += S[kk] * S[kk];
        score += 0.5f * (S_sq - sum_v_sq);

        // Sigmoid and threshold
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```