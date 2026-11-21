```hypothesis
Family: label-fm (Factorization Machine with 1st/2nd order terms).
Why: Adult has 14 mixed ordinal/categorical features; pairwise interactions (e.g., education×occupation, marital-status×sex) are known to boost accuracy over linear models. FMs capture all pairwise interactions with O(d·k) params (k=8 → ~32K weights) and O(d·k) inference, much cheaper than GBT/MLP.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000 (2× linear cost), the model doesn't justify its complexity.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
K_CHOICES = [4, 8, 12, 16]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def train_fm(X, y, k, rng, n_epochs):
    n, d = X.shape
    F = d * 256
    feat_idx = X.astype(np.int32) + np.arange(d) * 256  # (n, d)

    T1 = rng.normal(0, 0.01, F).astype(np.float32)
    T2 = rng.normal(0, 0.01, (F, k)).astype(np.float32)
    b = np.float32(0.0)

    m_T1, v_T1 = np.zeros_like(T1), np.zeros_like(T1)
    m_T2, v_T2 = np.zeros_like(T2), np.zeros_like(T2)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb_idx = feat_idx[idx]          # (B, d)
            yb = y[idx]                     # (B,)
            B = len(yb)

            # forward
            scores = b + np.sum(T1[xb_idx], axis=1)
            sum_v = np.zeros((B, k), np.float32)
            sum_v2 = np.zeros((B, k), np.float32)
            for jj in range(d):
                v = T2[xb_idx[:, jj]]
                sum_v += v
                sum_v2 += v * v
            scores += 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)

            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)  # (B,)

            # gradients on expanded features
            dT1 = np.zeros_like(T1)
            dT2 = np.zeros_like(T2)
            np.add.at(dT1, xb_idx.ravel(), np.repeat(grad, d))
            for jj in range(d):
                f_idx = xb_idx[:, jj]
                diff = sum_v - T2[f_idx]
                np.add.at(dT2, f_idx, grad[:, np.newaxis] * diff)
            dT1 /= B
            dT2 /= B

            # L2
            dT1 += L2 * T1
            dT2 += L2 * T2
            db = grad.mean() + L2 * b

            # Adam
            t += 1
            m_T1 = beta1 * m_T1 + (1 - beta1) * dT1
            v_T1 = beta2 * v_T1 + (1 - beta2) * dT1 * dT1
            T1 -= LR * (m_T1 / (1 - beta1 ** t)) / (np.sqrt(v_T1 / (1 - beta2 ** t)) + eps)

            m_T2 = beta1 * m_T2 + (1 - beta1) * dT2
            v_T2 = beta2 * v_T2 + (1 - beta2) * dT2 * dT2
            T2 -= LR * (m_T2 / (1 - beta1 ** t)) / (np.sqrt(v_T2 / (1 - beta2 ** t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db * db
            b -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    return T1.reshape(d, 256), T2.reshape(d, 256, k), b


def evaluate(X, y, T1, T2, b, k):
    n, d = X.shape
    scores = b + np.sum(T1[np.arange(d), X.T], axis=0)
    sum_v = np.zeros((n, k), np.float32)
    sum_v2 = np.zeros((n, k), np.float32)
    for j in range(d):
        v = T2[j, X[:, j], :]
        sum_v += v
        sum_v2 += v * v
    scores += 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
    p = sigmoid(scores)
    return np.mean((p > 0.5).astype(np.int64) == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    n_valid = n // 5
    n_train = n - n_valid

    X_train, X_valid = X[:n_train], X[n_train:]
    y_train, y_valid = y[:n_train], y[n_train:]

    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))

    best_k, best_acc = K_CHOICES[0], -1.0
    for k in K_CHOICES:
        T1, T2, b = train_fm(X_train, y_train, k, rng, n_epochs)
        acc = evaluate(X_valid, y_valid, T1, T2, b, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    T1, T2, b = train_fm(X, y, best_k, rng, n_epochs_full)

    # magic=0x464D3130 ("FM10"), d, k, C
    out = struct.pack("<4i", 0x464D3130, d, best_k, C)
    out += T1.astype("<f4").tobytes()
    out += T2.astype("<f4").tobytes()
    out += struct.pack("<f", b)
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
    if (C != 2 || d != 14 || k < 1 || k > 32) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 + (long)d * 256 * k + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 4);
    const float *T2 = T1 + (long)d * 256;
    const float *b_ptr = T2 + (long)d * 256 * k;
    float bias = *b_ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = bias;
        for (int j = 0; j < d; j++)
            score += T1[(long)j * 256 + xi[j]];

        memset(sum_v, 0, sizeof(float) * k);
        memset(sum_v2, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *v = T2 + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < k; f++)
            interaction += sum_v[f] * sum_v[f] - sum_v2[f];
        score += 0.5f * interaction;

        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(sum_v);
    free(sum_v2);
    return 0;
}
```