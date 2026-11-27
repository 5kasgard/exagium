```hypothesis
Family: label-fm (factorization machine). Adult income has strong feature interactions (e.g., education×occupation); FM captures these via low-rank embeddings with only O(d·k) extra inference cost. We tune latent dimension k∈{2,4,8,16} on the last 20% of train, then train on all data. Expect dev accuracy >0.8524 (label-linear) at <1000 Ir/item. Failure if accuracy ≤0.8524 or cost ≥1000 Ir/item.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 256
EPOCHS = 20
CANDIDATE_K = [2, 4, 8, 16]


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4 : i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_k = None
    best_acc = -1.0

    for k in CANDIDATE_K:
        rng = np.random.default_rng(SEED)
        linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
        emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
        bias = np.float32(0.0)

        m_linear = np.zeros_like(linear)
        v_linear = np.zeros_like(linear)
        m_emb = np.zeros_like(emb)
        v_emb = np.zeros_like(emb)
        m_bias = 0.0
        v_bias = 0.0
        t = 0

        n_train = X_train.shape[0]
        for _ in range(EPOCHS):
            order = rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s : s + BATCH]
                x = X_train[idx]
                yy = y_train[idx]
                B = len(idx)

                lin = np.sum(linear[np.arange(d), x.T], axis=0)
                emb_sum = np.sum(emb[np.arange(d), x.T, :], axis=0)
                emb_sq_sum = np.sum(emb[np.arange(d), x.T, :] ** 2, axis=0)
                inter = 0.5 * np.sum(emb_sum**2 - emb_sq_sum, axis=1)
                logit = lin + inter + bias

                p = 1.0 / (1.0 + np.exp(-logit))
                grad = p - yy

                grad_linear = np.zeros_like(linear)
                for j in range(d):
                    np.add.at(grad_linear[j], x[:, j], grad)
                grad_linear /= B
                grad_linear += L2 * linear

                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    vals = x[:, j]
                    sum_grad = np.zeros(256, dtype=np.float32)
                    np.add.at(sum_grad, vals, grad)
                    sum_grad_emb = np.zeros((256, k), dtype=np.float32)
                    for f in range(k):
                        np.add.at(sum_grad_emb[:, f], vals, grad * emb_sum[:, f])
                    grad_emb[j] = sum_grad_emb - emb[j] * sum_grad[:, None]
                grad_emb /= B
                grad_emb += L2 * emb

                grad_bias = np.mean(grad)

                t += 1
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8
                m_linear = beta1 * m_linear + (1 - beta1) * grad_linear
                v_linear = beta2 * v_linear + (1 - beta2) * grad_linear**2
                m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
                v_emb = beta2 * v_emb + (1 - beta2) * grad_emb**2
                m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
                v_bias = beta2 * v_bias + (1 - beta2) * grad_bias**2

                m_linear_hat = m_linear / (1 - beta1**t)
                v_linear_hat = v_linear / (1 - beta2**t)
                m_emb_hat = m_emb / (1 - beta1**t)
                v_emb_hat = v_emb / (1 - beta2**t)
                m_bias_hat = m_bias / (1 - beta1**t)
                v_bias_hat = v_bias / (1 - beta2**t)

                linear -= LR * m_linear_hat / (np.sqrt(v_linear_hat) + eps)
                emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)
                bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + eps)

        lin = np.sum(linear[np.arange(d), X_tune.T], axis=0)
        emb_sum = np.sum(emb[np.arange(d), X_tune.T, :], axis=0)
        emb_sq_sum = np.sum(emb[np.arange(d), X_tune.T, :] ** 2, axis=0)
        inter = 0.5 * np.sum(emb_sum**2 - emb_sq_sum, axis=1)
        logit = lin + inter + bias
        pred = (logit > 0).astype(np.int64)
        acc = np.mean(pred == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    # Retrain on full data with best_k
    k = best_k
    rng = np.random.default_rng(SEED)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.float32(0.0)

    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    m_bias = 0.0
    v_bias = 0.0
    t = 0

    n_full = X.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s : s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(idx)

            lin = np.sum(linear[np.arange(d), x.T], axis=0)
            emb_sum = np.sum(emb[np.arange(d), x.T, :], axis=0)
            emb_sq_sum = np.sum(emb[np.arange(d), x.T, :] ** 2, axis=0)
            inter = 0.5 * np.sum(emb_sum**2 - emb_sq_sum, axis=1)
            logit = lin + inter + bias

            p = 1.0 / (1.0 + np.exp(-logit))
            grad = p - yy

            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], x[:, j], grad)
            grad_linear /= B
            grad_linear += L2 * linear

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                vals = x[:, j]
                sum_grad = np.zeros(256, dtype=np.float32)
                np.add.at(sum_grad, vals, grad)
                sum_grad_emb = np.zeros((256, k), dtype=np.float32)
                for f in range(k):
                    np.add.at(sum_grad_emb[:, f], vals, grad * emb_sum[:, f])
                grad_emb[j] = sum_grad_emb - emb[j] * sum_grad[:, None]
            grad_emb /= B
            grad_emb += L2 * emb

            grad_bias = np.mean(grad)

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_linear = beta1 * m_linear + (1 - beta1) * grad_linear
            v_linear = beta2 * v_linear + (1 - beta2) * grad_linear**2
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb**2
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias**2

            m_linear_hat = m_linear / (1 - beta1**t)
            v_linear_hat = v_linear / (1 - beta2**t)
            m_emb_hat = m_emb / (1 - beta1**t)
            v_emb_hat = v_emb / (1 - beta2**t)
            m_bias_hat = m_bias / (1 - beta1**t)
            v_bias_hat = v_bias / (1 - beta2**t)

            linear -= LR * m_linear_hat / (np.sqrt(v_linear_hat) + eps)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)
            bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + eps)

    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, k, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += struct.pack("<f", bias)
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 256 + 4L * d * 256 * k + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + d * 256;
    const float bias = *(const float *)(emb + d * 256 * k);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float linear_sum = 0.0f;
        float emb_sum[64];
        float emb_sq_sum[64];
        for (int f = 0; f < k; f++) { emb_sum[f] = 0.0f; emb_sq_sum[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            linear_sum += linear[j * 256 + v];
            const float *e = emb + (j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                emb_sum[f] += val;
                emb_sq_sum[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) {
            inter +=