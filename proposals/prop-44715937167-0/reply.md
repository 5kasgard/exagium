```hypothesis
Family: label-fm (factorization machine). Treats each of the 14 bytes as a categorical field (256 values) and learns a linear weight plus a low-dimensional embedding per field-value pair. Pairwise interactions are captured via the FM trick (sum^2 - sum of squares) with a tiny projection to classes. Should beat label-linear on Adult because it models feature crosses without blowing up parameters, and stays fast: only O(d*k + k*C) extra ops per item. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the idea fails.
```
```python
"""label-fm: factorization machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each byte as a field index (0..d-1) and value (0..255). Learns:
  bias[C], linear[d][256][C], emb[d][256][k], V[k][C].
Score_c = bias_c + sum_i linear[i][x_i][c] + 0.5 * ( (sum_i emb[i][x_i])^2 - sum_i emb[i][x_i]^2 ) @ V[:,c]
Hyperparameter k chosen on last 20% of train.in (file order), then retrained on all train."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8, 16]  # embedding sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, C, k, rng, n_epochs):
    n, d = X.shape
    # params: bias[C], linear[d,256,C], emb[d,256,k], V[k,C]
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-0.01, 0.01, (d, 256, C)) * math.sqrt(6.0 / (256 * C))).astype(np.float32)
    emb = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / (256 * k))).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (k, C)) * math.sqrt(6.0 / (k * C))).astype(np.float32)
    params = [bias, linear, emb, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(idx):
        xb = X[idx]  # (B, d)
        B = len(idx)
        # linear part
        lin = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B, C)
        # FM interaction
        e = emb[np.arange(d), xb.T]  # (d, B, k)
        sum_e = np.sum(e, axis=0)  # (B, k)
        sum_e2 = np.sum(e * e, axis=0)
        inter = 0.5 * (sum_e * sum_e - sum_e2)  # (B, k)
        scores = lin + inter @ V  # (B, C)
        return scores, sum_e, inter

    def softmax_grad(scores, yy):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs / len(yy)  # (B, C)

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            scores, sum_e, inter = forward(idx)
            g = softmax_grad(scores, yy)  # (B, C)

            # gradients
            g_bias = g.sum(axis=0)  # (C,)
            g_linear = np.zeros_like(linear)
            for j in range(d):
                vals = X[idx, j]
                np.add.at(g_linear[j], vals, g)
            # emb grad: dL/demb = (g @ V.T) * (sum_e - emb)  [per feature]
            g_inter = g @ V.T  # (B, k)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = X[idx, j]
                # for each sample, contribution to feature j is g_inter[b] * (sum_e[b] - emb[j, vals[b]])
                diff = sum_e - emb[j, vals]  # (B, k)
                grad_emb_j = g_inter * diff  # (B, k)
                np.add.at(g_emb[j], vals, grad_emb_j)
            g_V = inter.T @ g  # (k, C)

            # L2 regularization
            g_linear += L2 * linear
            g_emb += L2 * emb
            g_V += L2 * V

            grads = [g_bias, g_linear, g_emb, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return bias, linear, emb, V


def eval_acc(X, y, C, bias, linear, emb, V):
    scores, _, _ = forward_fm(X, bias, linear, emb, V)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def forward_fm(X, bias, linear, emb, V):
    n, d = X.shape
    lin = bias + np.sum(linear[np.arange(d), X.T], axis=0)
    e = emb[np.arange(d), X.T]
    sum_e = np.sum(e, axis=0)
    sum_e2 = np.sum(e * e, axis=0)
    inter = 0.5 * (sum_e * sum_e - sum_e2)
    scores = lin + inter @ V
    return scores, sum_e, inter


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_k, best_acc = None, -1.0
    for k in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + k)  # deterministic per k
        bias, linear, emb, V = train_fm(X_train, y_train, C, k, rng_k, n_epochs)
        acc = eval_acc(X_tune, y_tune, C, bias, linear, emb, V)
        if acc > best_acc:
            best_acc, best_k = acc, k
            best_params = (bias, linear, emb, V)

    # retrain on full data with best_k
    rng_full = np.random.default_rng(SEED + best_k * 100)
    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, linear, emb, V = train_fm(X, y, C, best_k, rng_full, n_epochs_full)

    # write weights.bin: magic, d, k, C; then bias, linear, emb, V (all float32 little-endian)
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index per record.
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 bias[C], linear[d*256*C], emb[d*256*k], V[k*C]. */
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
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * k + (long)k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * 256 * C;
    const float *V = emb + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = (float *)malloc(sizeof(float) * k);
    float *sum_e2 = (float *)malloc(sizeof(float) * k);
    float *inter = (float *)malloc(sizeof(float) * k);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // FM interaction
        for (int u = 0; u < k; u++) { sum_e[u] = 0.0f; sum_e2[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_e[u] += val;
                sum_e2[u] += val * val;
            }
        }
        for (int u = 0; u < k; u++) inter[u] = 0.5f * (sum_e[u] * sum_e[u] - sum_e2[u]);
        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int u = 0; u < k; u++) acc += inter[u] * V[(long)u * C + c];
            scores[c] += acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```