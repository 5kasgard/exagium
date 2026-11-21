```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise pixel interactions via low-rank embeddings (k=1..4), crucial for digit strokes, beating linear (0.961) at ~3-5x CPU cost. Tuning selects k on last 20% of train. Failure if k=1,2 underfits or overfits small data (2867 samples).
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [1, 2, 4]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, C, k, rng, n_epochs):
    n, d = X.shape
    linear = np.zeros((d, 17, C), np.float32)
    emb = (rng.normal(0, 0.01, (d, 17, C, k))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [linear, emb, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            lin_scores = np.zeros((B, C), np.float32)
            for j in range(d):
                vals = xb[:, j]
                lin_scores += linear[j, vals, :]
            lin_scores += bias

            sum_emb = np.zeros((B, C, k), np.float32)
            sum_emb_sq = np.zeros((B, C, k), np.float32)
            for j in range(d):
                vals = xb[:, j]
                e = emb[j, vals, :, :]
                sum_emb += e
                sum_emb_sq += e * e

            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=2)
            scores = lin_scores + inter

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B

            grad_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    np.add.at(grad_linear[j, :, c], vals, grad_scores[:, c])
            grad_linear += L2 * linear

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    g = grad_scores[:, c]
                    diff = sum_emb[:, c, :] - emb[j, vals, c, :]
                    for m in range(k):
                        np.add.at(grad_emb[j, :, c, m], vals, g * diff[:, m])
            grad_emb += L2 * emb

            grad_bias = grad_scores.sum(axis=0) + L2 * bias

            grads = [grad_linear, grad_emb, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return linear, emb, bias


def evaluate_fm(X, y, linear, emb, bias, C, k):
    n, d = X.shape
    lin_scores = np.zeros((n, C), np.float32)
    for j in range(d):
        vals = X[:, j]
        lin_scores += linear[j, vals, :]
    lin_scores += bias

    sum_emb = np.zeros((n, C, k), np.float32)
    sum_emb_sq = np.zeros((n, C, k), np.float32)
    for j in range(d):
        vals = X[:, j]
        e = emb[j, vals, :, :]
        sum_emb += e
        sum_emb_sq += e * e

    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=2)
    scores = lin_scores + inter
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    n = len(X_all)
    split = n - n // 5
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    best_k = 1
    best_acc = -1.0
    n_epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for k in K_CHOICES:
        r = np.random.default_rng(SEED + k)
        linear, emb, bias = train_fm(X_train, y_train, C, k, r, n_epochs_tune)
        acc = evaluate_fm(X_tune, y_tune, linear, emb, bias, C, k)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    r = np.random.default_rng(SEED + best_k)
    linear, emb, bias = train_fm(X_all, y_all, C, best_k, r, n_epochs_full)

    linear_full = np.zeros((d, 256, C), np.float32)
    linear_full[:, :17, :] = linear
    emb_full = np.zeros((d, 256, C, best_k), np.float32)
    emb_full[:, :17, :, :] = emb

    out = struct.pack("<4i", 0x464D31, d, best_k, C)
    out += linear_full.astype("<f4").tobytes()
    out += emb_full.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()

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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d * 256 * C * (1 + k) + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (long)d * 256 * C;
    const float *bias = emb + (long)d * 256 * C * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * C * k);
    float *sum_emb_sq = malloc(sizeof(float) * C * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias[c];
        for (int c = 0; c < C; c++) for (int m = 0; m < k; m++) {
            sum_emb[c * k + m] = 0.0f;
            sum_emb_sq[c * k + m] = 0.0f;
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_row = linear + ((long)j * 256 + v) * C;
            const float *emb_row = emb + ((long)j * 256 + v) * C * k;
            for (int c = 0; c < C; c++) {
                acc[c] += lin_row[c];
                for (int m = 0; m < k; m++) {
                    float e = emb_row[c * k + m];
                    sum_emb[c * k + m] += e;
                    sum_emb_sq[c * k + m] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int m = 0; m < k; m++) {
                float s = sum_emb[c * k + m];
                inter += s * s - sum_emb_sq[c * k + m];
            }
            acc[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```