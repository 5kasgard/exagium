```hypothesis
Family: label-fm (factorization machine with per-field embeddings).
Why: Adult has 14 fields each quantized to 256 values; a field-aware FM captures pairwise interactions with only 14*256*k extra parameters (k=8 → ~32k floats) and 91 dot products per item. This should beat the linear baseline (0.8524) while staying far cheaper than GBDT/MLP.
Failure signal: dev accuracy ≤ 0.8524 or inference cost > 2× label-linear.
```
```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 fields as a 256-value embedding (k=8 by default, tuned on last 20% of train.in).
Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 0.01, 1e-4, 64, 600000, 50
K_CANDIDATES = [4, 8, 16]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_val(X, y, val_frac=0.2):
    n = X.shape[0]
    n_val = int(n * val_frac)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def train_fm(X, y, k, rng, epochs, batch_size):
    n, d = X.shape
    C = 2
    # parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]
            # forward
            lin_part = linear[np.arange(d), xb.T].sum(axis=0)  # (b,)
            emb_vals = embed[np.arange(d), xb.T]               # (d, b, k)
            sum_emb = emb_vals.sum(axis=0)                     # (b, k)
            sum_emb_sq = (emb_vals ** 2).sum(axis=0)           # (b, k)
            inter = 0.5 * ((sum_emb ** 2 - sum_emb_sq).sum(axis=1))  # (b,)
            logit = bias + lin_part + inter
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)   # (b,)
            # gradients
            dbias = grad.mean()
            # linear grad: for each feature j, value v, add mean grad over examples where xb[:,j]==v
            dlin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        dlin[j, v] = grad[mask].mean()
            # embedding grad: dL/de_{j,l} = grad * (sum_emb_l - e_{j,l})
            demb = np.zeros_like(embed)
            for j in range(d):
                e = emb_vals[j]          # (b, k)
                factor = (grad[:, None] * (sum_emb - e)).mean(axis=0)  # (k,)
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        demb[j, v] = (grad[mask, None] * (sum_emb[mask] - e[mask])).mean(axis=0)
            # L2 regularization
            dlin += L2 * linear
            demb += L2 * embed
            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * dbias
            v_bias = beta2 * v_bias + (1 - beta2) * dbias * dbias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * dlin
            v_lin = beta2 * v_lin + (1 - beta2) * dlin * dlin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * demb
            v_emb = beta2 * v_emb + (1 - beta2) * demb * demb
            embed -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return bias, linear, embed


def evaluate(X, y, bias, linear, embed):
    n, d = X.shape
    k = embed.shape[2]
    lin_part = linear[np.arange(d), X.T].sum(axis=0)
    emb_vals = embed[np.arange(d), X.T]
    sum_emb = emb_vals.sum(axis=0)
    sum_emb_sq = (emb_vals ** 2).sum(axis=0)
    inter = 0.5 * ((sum_emb ** 2 - sum_emb_sq).sum(axis=1))
    logit = bias + lin_part + inter
    pred = (logit > 0).astype(np.int64)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # tune k on last 20% of train.in (by file order)
    X_tr, y_tr, X_val, y_val = split_train_val(X, y, 0.2)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X_tr.shape[0]))
    for k in K_CANDIDATES:
        bias, linear, embed = train_fm(X_tr, y_tr, k, rng, epochs, BATCH)
        acc = evaluate(X_val, y_val, bias, linear, embed)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # retrain on all data with best_k
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))
    bias, linear, embed = train_fm(X, y, best_k, rng, epochs, BATCH)

    # write weights.bin
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, best_k, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Each input record = d bytes (one per field). Output = 1 byte class (0/1).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias, linear[d][256], embed[d][256][k]. */
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
    if (C != 2 || d < 1 || k < 1 || k > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float logit = bias;
        // linear part
        for (int j = 0; j < d; j++) logit += linear[j * 256 + xi[j]];
        // interaction part: 0.5 * sum_l ( (sum_j e_{j,l})^2 - sum_j e_{j,l}^2 )
        memset(sum_emb, 0, sizeof(float) * k);
        memset(sum_emb_sq, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * k;
            for (int l = 0; l < k; l++) {
                float v = e[l];
                sum_emb[l] += v;
                sum_emb_sq[l] += v * v;
            }
        }
        float inter = 0.0f;
        for (int l = 0; l < k; l++) {
            float s = sum_emb[l];
            inter += 0.5f * (s * s - sum_emb_sq[l]);
        }
        logit += inter;

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```