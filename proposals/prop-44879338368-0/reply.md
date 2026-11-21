```hypothesis
Factorization Machine (FM) with low-rank pairwise interactions. The Adult dataset is known to benefit from feature crosses (e.g., education×occupation). An FM captures all pairwise interactions via latent vectors of size K, adding only O(d*K) parameters and O(d*K) inference cost. With K=8–32 it should surpass the linear baseline (0.8524) while staying fast. If dev accuracy ≤0.8524 or instructions/item exceed label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
BATCH = 64
MAX_EPOCHS_TUNE = 5
MAX_EPOCHS_FINAL = 20
K_CANDIDATES = [4, 8, 16, 32]
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def forward(x_batch, linear, latent, bias):
    B, d = x_batch.shape
    K = latent.shape[2]
    linear_sum = np.zeros(B, dtype=np.float32)
    s = np.zeros((B, K), dtype=np.float32)
    v_sq_sum = np.zeros(B, dtype=np.float32)
    for j in range(d):
        idx = x_batch[:, j]
        linear_sum += linear[j, idx]
        v_j = latent[j, idx]
        s += v_j
        v_sq_sum += np.sum(v_j * v_j, axis=1)
    interaction = 0.5 * (np.sum(s * s, axis=1) - v_sq_sum)
    logit = bias + linear_sum + interaction
    return logit, s


def evaluate(X, y, indices, linear, latent, bias):
    x_val = X[indices]
    y_val = y[indices]
    logit, _ = forward(x_val, linear, latent, bias)
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y_val)


def train_loop(X, y, idx, linear, latent, bias, epochs, rng):
    n = len(idx)
    d, _, K = latent.shape
    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_latent = np.zeros_like(latent)
    v_latent = np.zeros_like(latent)
    m_bias = 0.0
    v_bias = 0.0
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = idx[order[s:s + BATCH]]
            x_batch = X[batch_idx]
            y_batch = y[batch_idx]
            logit, s_vec = forward(x_batch, linear, latent, bias)
            prob = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = prob - y_batch
            dbias = np.mean(grad_logit)
            grad_linear = np.zeros_like(linear)
            grad_latent = np.zeros_like(latent)
            for j in range(d):
                idx_j = x_batch[:, j]
                np.add.at(grad_linear[j], idx_j, grad_logit)
                v_j = latent[j, idx_j]
                diff = s_vec - v_j
                g = grad_logit[:, np.newaxis] * diff
                np.add.at(grad_latent[j], idx_j, g)
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * dbias
            v_bias = BETA2 * v_bias + (1 - BETA2) * dbias * dbias
            bias -= LR * (m_bias / (1 - BETA1 ** t)) / (np.sqrt(v_bias / (1 - BETA2 ** t)) + EPS)
            # linear
            m_linear *= BETA1
            m_linear += (1 - BETA1) * grad_linear
            v_linear *= BETA2
            v_linear += (1 - BETA2) * grad_linear * grad_linear
            linear -= LR * (m_linear / (1 - BETA1 ** t)) / (np.sqrt(v_linear / (1 - BETA2 ** t)) + EPS)
            # latent
            m_latent *= BETA1
            m_latent += (1 - BETA1) * grad_latent
            v_latent *= BETA2
            v_latent += (1 - BETA2) * grad_latent * grad_latent
            latent -= LR * (m_latent / (1 - BETA1 ** t)) / (np.sqrt(v_latent / (1 - BETA2 ** t)) + EPS)
    return bias, linear, latent


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    best_K = K_CANDIDATES[0]
    best_acc = -1.0
    for K in K_CANDIDATES:
        rng = np.random.default_rng(SEED + K)
        linear = np.zeros((d, 256), dtype=np.float32)
        latent = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
        bias = np.float32(0.0)
        bias, linear, latent = train_loop(X, y, train_idx, linear, latent, bias, MAX_EPOCHS_TUNE, rng)
        acc = evaluate(X, y, val_idx, linear, latent, bias)
        if acc > best_acc:
            best_acc = acc
            best_K = K

    rng = np.random.default_rng(SEED)
    linear = np.zeros((d, 256), dtype=np.float32)
    latent = rng.normal(0, 0.01, size=(d, 256, best_K)).astype(np.float32)
    bias = np.float32(0.0)
    bias, linear, latent = train_loop(X, y, np.arange(n), linear, latent, bias, MAX_EPOCHS_FINAL, rng)

    magic = 0x464D5442  # "BTMF" little-endian
    out = struct.pack("<4i", magic, d, best_K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D5442) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected_floats = 1L + (long)d * 256 * (1L + K);
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias_ptr = (const float *)(hd + 4);
    const float *linear = bias_ptr + 1;
    const float *latent = linear + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = 0.0f;
        float v_sq_sum = 0.0f;
        memset(s, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            uint8_t val = xi[j];
            linear_sum += linear[(long)j * 256 + val];
            const float *v = latent + ((long)j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float vk = v[kk];
                s[kk] += vk;
                v_sq_sum += vk * vk;
            }
        }
        float dot_s = 0.0f;
        for (int kk = 0; kk < K; kk++) dot_s += s[kk] * s[kk];
        float interaction = 0.5f * (dot_s - v_sq_sum);
        float logit = bias_ptr[0] + linear_sum + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```