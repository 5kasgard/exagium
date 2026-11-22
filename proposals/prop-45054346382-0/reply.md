```hypothesis
Family: label-fm (factorization machine with shared embeddings, per-class linear terms).
Why: Adult has strong categorical interactions (e.g., education × occupation). A degree-2 FM captures pairwise interactions with O(d·k) parameters and inference cost (k=4), far cheaper than GBDT/MLP while adding non-linearity.
Failure: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 719 Ir/item.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 input bytes as a categorical field (256 values).
Learns per-class linear weights (C x 14 x 256) and shared embeddings (14 x 256 x k).
Interaction term: 0.5 * ( (sum emb)^2 - sum emb^2 ). Adam, L2, minibatch.
Tunes k in {2,4,8} and lr in {0.01,0.03} on last 20% of train.in (5 epochs),
then retrains on full train with best config for full epochs.
Weights format: magic 0x464D464D, d, k, C, version; then float32 arrays:
  linear[C][d][256], emb[d][256][k], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4
TUNE_EPOCHS = 5
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, C, d, k, lr, epochs, rng):
    n = X.shape[0]
    # params
    linear = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    # adam state
    m_linear, v_linear = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)
            # linear scores: (B, C)
            lin_scores = bias + np.sum(linear[:, np.arange(d), xb.T], axis=2).T
            # embeddings: (B, d, k)
            emb_b = emb[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_emb = emb_b.sum(axis=1)                 # (B, k)
            sum_emb_sq = (emb_b ** 2).sum(axis=(1, 2))  # (B,)
            interaction = 0.5 * ((sum_emb ** 2).sum(axis=1) - sum_emb_sq)  # (B,)
            scores = lin_scores + interaction[:, None]  # (B, C)
            # softmax + grad
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B  # (B, C)
            g_sum = grad_scores.sum(axis=1)  # (B,)
            # gradients
            # bias
            g_bias = grad_scores.sum(axis=0)
            # linear: for each class, field, value
            g_linear = np.zeros_like(linear)
            for c in range(C):
                for j in range(d):
                    vals = xb[:, j]
                    np.add.at(g_linear[c, j], vals, grad_scores[:, c])
            # emb: (d, 256, k)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                v_j = emb_b[:, j, :]  # (B, k)
                # g_emb[j, vals] += g_sum[:, None] * (sum_emb - v_j)
                diff = g_sum[:, None] * (sum_emb - v_j)  # (B, k)
                for dim in range(k):
                    np.add.at(g_emb[j, vals, dim], np.arange(B), diff[:, dim])
            # L2
            g_linear += L2 * linear
            g_emb += L2 * emb
            g_bias += L2 * bias
            # adam update
            t += 1
            for param, grad, m, v in [
                (linear, g_linear, m_linear, v_linear),
                (emb, g_emb, m_emb, v_emb),
                (bias, g_bias, m_bias, v_bias),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return linear, emb, bias


def accuracy(X, y, C, d, k, linear, emb, bias):
    n = X.shape[0]
    lin_scores = bias + np.sum(linear[:, np.arange(d), X.T], axis=2).T
    emb_b = emb[np.arange(d), X.T].transpose(1, 0, 2)
    sum_emb = emb_b.sum(axis=1)
    sum_emb_sq = (emb_b ** 2).sum(axis=(1, 2))
    interaction = 0.5 * ((sum_emb ** 2).sum(axis=1) - sum_emb_sq)
    scores = lin_scores + interaction[:, None]
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_acc = -1.0
    best_k, best_lr = 4, 0.01
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            r = np.random.default_rng(SEED)
            lin, emb, bias = train_fm(X_tr, y_tr, C, d, k, lr, TUNE_EPOCHS, r)
            acc = accuracy(X_val, y_val, C, d, k, lin, emb, bias)
            if acc > best_acc:
                best_acc, best_k, best_lr = acc, k, lr
    # final training on all data
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    r = np.random.default_rng(SEED)
    linear, emb, bias = train_fm(X, y, C, d, best_k, best_lr, epochs, r)
    # write weights
    magic = 0x464D464D  # 'FMFM'
    version = 1
    header = struct.pack("<5i", magic, d, best_k, C, version)
    out = header
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weights format (little-endian):
 *   int32 magic=0x464D464D, d, k, C, version
 *   float32 linear[C][d][256]
 *   float32 emb[d][256][k]
 *   float32 bias[C] */
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
    if (nw < 20 || hd[0] != 0x464D464D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3], version = hd[4];
    if (d != 14 || C != 2 || k < 1 || k > 8) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t n_linear = (size_t)C * d * 256;
    size_t n_emb = (size_t)d * 256 * k;
    size_t n_bias = C;
    size_t total_floats = n_linear + n_emb + n_bias;
    if (nw != 20 + 4L * (long)total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 5);
    const float *linear = p; p += n_linear;
    const float *emb = p; p += n_emb;
    const float *bias = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score[2] = {bias[0], bias[1]};
        memset(sum_emb, 0, sizeof(float) * k);
        float sum_emb_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_j = linear + (size_t)j * 256;
            for (int c = 0; c < C; c++) score[c] += lin_j[c * d * 256 + v];
            const float *e = emb + ((size_t)j * 256 + v) * k;
            for (int dim = 0; dim < k; dim++) {
                float val = e[dim];
                sum_emb[dim] += val;
                sum_emb_sq += val * val;
            }
        }
        float interaction = 0.0f;
        for (int dim = 0; dim < k; dim++) interaction += sum_emb[dim] * sum_emb[dim];
        interaction = 0.5f * (interaction - sum_emb_sq);
        score[0] += interaction; score[1] += interaction;
        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```