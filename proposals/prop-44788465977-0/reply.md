```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong feature interactions (education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via factorized latent vectors (k=8) with O(d·k) inference using the kernel trick, keeping cost near label-linear. If dev accuracy ≤ 0.8524 (label-linear) or latency ≥ 2× label-linear, the approach fails.
```
```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Binary FM: logit = w0 + sum w_i[x_i] + 0.5*(||sum v_i[x_i]||^2 - sum ||v_i[x_i]||^2).
Latent dim k=8. Adam lr=1e-3, L2=1e-5, batch=256, epochs=min(50, ceil(500000/n)).
Weights: int32 magic 0x464D3131, d, k, C; float32 w0, w[d][256], v[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-5, 256, 500000, 50
MAGIC = 0x464D3131  # "FM11"

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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: w0 (scalar), w[d,256], v[d,256,k]
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_v = np.zeros_like(v); v_v = np.zeros_like(v)

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]  # (B, d)
            yb = y[idx]  # (B,)
            B = len(idx)

            # Forward
            # Linear part: sum_j w[j, Xb[:,j]]
            lin = np.zeros(B, np.float32)
            for j in range(d):
                lin += w[j, Xb[:, j]]

            # Latent sum S = sum_j v[j, Xb[:,j], :] -> (B, K)
            S = np.zeros((B, K), np.float32)
            for j in range(d):
                S += v[j, Xb[:, j], :]

            # Interaction: 0.5 * (||S||^2 - sum_j ||v_j||^2)
            S_sq = np.sum(S * S, axis=1)  # (B,)
            v_sq = np.zeros(B, np.float32)
            for j in range(d):
                v_sq += np.sum(v[j, Xb[:, j], :] ** 2, axis=1)
            inter = 0.5 * (S_sq - v_sq)

            logit = w0 + lin + inter  # (B,)
            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-logit))
            # Gradient of BCE: prob - y
            grad = (prob - yb).astype(np.float32)  # (B,)

            # Backward
            # w0
            gw0 = grad.mean()
            # w[j, val]
            gw = np.zeros_like(w)
            for j in range(d):
                for b in range(B):
                    gw[j, Xb[b, j]] += grad[b]
            gw /= B
            # v[j, val, k]: grad * (S - v_j)
            gv = np.zeros_like(v)
            for j in range(d):
                for b in range(B):
                    val = Xb[b, j]
                    gv[j, val, :] += grad[b] * (S[b, :] - v[j, val, :])
            gv /= B

            # L2 regularization
            gw += L2 * w
            gv += L2 * v

            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * gv * gv
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # Pack weights: magic, d, k, C; w0; w[d][256]; v[d][256][k]
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (binary) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (feature values 0-255). Output: 1 byte class (0/1).
 * Weights: int32 magic 0x464D3131, d, k, C; float32 w0, w[d][256], v[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || d > 256 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;                    // [d][256]
    const float *v = p + 1 + d * 256;          // [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = (float *)malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin = 0.0f;
        memset(S, 0, sizeof(float) * K);
        float v_sq_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            lin += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * K;
            for (int u = 0; u < K; u++) {
                float vju = vj[u];
                S[u] += vju;
                v_sq_sum += vju * vju;
            }
        }
        float S_sq = 0.0f;
        for (int u = 0; u < K; u++) S_sq += S[u] * S[u];
        float inter = 0.5f * (S_sq - v_sq_sum);
        float logit = w0 + lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(S);
    return 0;
}
```