```hypothesis
Family: label-fm (Factorization Machine). Adult income depends on feature interactions (e.g., education × occupation, marital-status × hours-per-week) that linear models miss. A field-aware FM with small embeddings (k=8) adds only 91 dot-products per record, keeping CPU cost near label-linear while capturing pairwise interactions. If dev accuracy ≤ 0.8524 or ns/item > 100, the approach fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
SAMPLES = 600000
CANDIDATE_K = [4, 8, 16, 32]


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_fm(X, y, d, k, lr, l2, batch, epochs, rng):
    n = X.shape[0]
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)

    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            yb = y[idx]
            b = len(idx)

            logits = np.full(b, bias, np.float32)
            for j in range(d):
                logits += linear[j, xb[:, j]]

            embs = emb[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_emb = embs.sum(axis=1)
            sum_emb_sq = (embs * embs).sum(axis=1)
            inter = 0.5 * ((sum_emb * sum_emb) - sum_emb_sq).sum(axis=1)
            logits += inter

            p = sigmoid(logits)
            grad_logit = (p - yb).astype(np.float32)

            g_bias = grad_logit.mean()
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad_logit)
            g_lin /= b

            g_emb = np.zeros_like(emb)
            other_sum = sum_emb[:, None, :] - embs
            grad_emb = grad_logit[:, None, None] * other_sum
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], grad_emb[:, j, :])
            g_emb /= b

            g_lin += l2 * linear
            g_emb += l2 * emb
            g_bias += l2 * bias

            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return bias, linear, emb


def evaluate(X, y, bias, linear, emb):
    n = X.shape[0]
    logits = np.full(n, bias, np.float32)
    d = X.shape[1]
    for j in range(d):
        logits += linear[j, X[:, j]]
    embs = emb[np.arange(d), X.T].transpose(1, 0, 2)
    sum_emb = embs.sum(axis=1)
    sum_emb_sq = (embs * embs).sum(axis=1)
    inter = 0.5 * ((sum_emb * sum_emb) - sum_emb_sq).sum(axis=1)
    logits += inter
    preds = (logits > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_k = 8
    best_acc = 0.0
    for k in CANDIDATE_K:
        bias, linear, emb = train_fm(X_train, y_train, d, k, LR, L2, BATCH, MAX_EPOCHS, rng)
        acc = evaluate(X_val, y_val, bias, linear, emb)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    bias, linear, emb = train_fm(X, y, d, best_k, LR, L2, BATCH, MAX_EPOCHS, rng)

    out = struct.pack("<4i", 0x464D3130, d, best_k, 2)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k < 1 || k > 64 || d != 14) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float logit = bias;
        for (int j = 0; j < d; j++)
            logit += linear[(long)j * 256 + xi[j]];

        float sum_emb[64], sum_emb_sq[64];
        for (int u = 0; u < k; u++) {
            sum_emb[u] = 0.0f;
            sum_emb_sq[u] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_emb_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++)
            inter += 0.5f * (sum_emb[u] * sum_emb[u] - sum_emb_sq[u]);
        logit += inter;

        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```