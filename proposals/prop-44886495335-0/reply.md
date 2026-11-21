```hypothesis
Family: label-fm (factorization machine with learned embeddings per feature value).
Why: Adult income prediction benefits from feature interactions (e.g., education × occupation). Label-linear only uses main effects; a degree-2 FM captures pairwise interactions via low-rank embeddings, adding only ~14*256*k parameters (k=8 → ~29k floats). Training with Adam on the tuning split (last 20% of train.in) should yield higher accuracy than label-linear (0.8524) at similar inference cost (O(d*k) per item). If dev accuracy does not exceed label-linear, it shows interactions are not decisive for this task.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the 14 bytes is treated as a categorical value in 0..255.
Model: logit = bias + sum linear[feat*256+val] + 0.5*( (sum embed)^2 - sum embed^2 ).
Hyperparameters (embedding size k, learning rate) chosen on last 20% of train.in, then retrained on all.
Exported weights: magic 0x464D4C31, d, k, C; float32 bias, linear[d*256], embed[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
EPOCHS_TUNE = 5
EPOCHS_FULL = 20
SAMPLES_PER_EPOCH = 600000  # not used directly; we use fixed epochs


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]

    # flat indices: for each feature j, value v -> index = j*256 + v
    idx = X + np.arange(d, dtype=np.uint16) * 256  # shape (n, d)

    # split: last 20% for tuning
    n_tune = n // 5
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)

    # hyperparameter grid
    k_options = [4, 8]
    lr_options = [1e-3, 3e-3]
    best_acc = -1.0
    best_params = None
    best_k = None
    best_lr = None

    rng = np.random.default_rng(SEED)

    for k in k_options:
        for lr in lr_options:
            # train on train_idx
            bias = 0.0
            linear = np.zeros(d * 256, dtype=np.float32)
            embed = (rng.normal(0, 0.01, size=(d * 256, k))).astype(np.float32)

            # Adam state
            m_bias, v_bias = 0.0, 0.0
            m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
            m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
            t = 0

            for _ in range(EPOCHS_TUNE):
                order = rng.permutation(len(train_idx))
                for s in range(0, len(train_idx), BATCH):
                    batch_idx = train_idx[order[s:s+BATCH]]
                    if len(batch_idx) == 0:
                        continue
                    batch_i = idx[batch_idx]
                    batch_y = y[batch_idx]

                    # forward
                    logits = np.full(len(batch_y), bias, dtype=np.float32)
                    sum_v = np.zeros((len(batch_y), k), dtype=np.float32)
                    sum_v2 = np.zeros((len(batch_y), k), dtype=np.float32)
                    for j in range(d):
                        col = batch_i[:, j]
                        logits += linear[col]
                        sum_v += embed[col]
                        sum_v2 += embed[col] * embed[col]
                    interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
                    logits += interaction

                    p = 1.0 / (1.0 + np.exp(-logits))
                    g = p - batch_y  # gradient of BCE w.r.t logit

                    # gradients
                    grad_bias = np.mean(g)
                    grad_lin = np.zeros_like(linear)
                    grad_emb = np.zeros_like(embed)

                    for j in range(d):
                        col = batch_i[:, j]
                        # linear gradient
                        np.add.at(grad_lin, col, g)
                        # embedding gradient: g * (sum_v - v)
                        # for each sample, for each feature, add g_i * (sum_v_i - v_ij)
                        # we can vectorize by looping over samples in batch (small)
                        for b in range(len(batch_y)):
                            v_ij = embed[col[b]]
                            grad_emb[col[b]] += g[b] * (sum_v[b] - v_ij)

                    grad_lin /= len(batch_y)
                    grad_emb /= len(batch_y)

                    # Adam update
                    t += 1
                    beta1, beta2 = 0.9, 0.999
                    eps = 1e-8

                    m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
                    v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias * grad_bias)
                    bias -= lr * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)

                    m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
                    v_lin = beta2 * v_lin + (1 - beta2) * (grad_lin * grad_lin)
                    linear -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

                    m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
                    v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
                    embed -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

            # evaluate on tune_idx
            tune_i = idx[tune_idx]
            tune_y = y[tune_idx]
            logits = np.full(len(tune_y), bias, dtype=np.float32)
            sum_v = np.zeros((len(tune_y), k), dtype=np.float32)
            sum_v2 = np.zeros((len(tune_y), k), dtype=np.float32)
            for j in range(d):
                col = tune_i[:, j]
                logits += linear[col]
                sum_v += embed[col]
                sum_v2 += embed[col] * embed[col]
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            logits += interaction
            pred = (logits > 0).astype(np.int64)
            acc = np.mean(pred == tune_y)
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_lr = lr
                best_params = (bias.copy(), linear.copy(), embed.copy())

    # retrain on all data with best hyperparameters
    k = best_k
    lr = best_lr
    bias, linear, embed = best_params
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility

    for _ in range(EPOCHS_FULL):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s+BATCH]
            if len(batch_idx) == 0:
                continue
            batch_i = idx[batch_idx]
            batch_y = y[batch_idx]

            logits = np.full(len(batch_y), bias, dtype=np.float32)
            sum_v = np.zeros((len(batch_y), k), dtype=np.float32)
            sum_v2 = np.zeros((len(batch_y), k), dtype=np.float32)
            for j in range(d):
                col = batch_i[:, j]
                logits += linear[col]
                sum_v += embed[col]
                sum_v2 += embed[col] * embed[col]
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            logits += interaction

            p = 1.0 / (1.0 + np.exp(-logits))
            g = p - batch_y

            grad_bias = np.mean(g)
            grad_lin = np.zeros_like(linear)
            grad_emb = np.zeros_like(embed)

            for j in range(d):
                col = batch_i[:, j]
                np.add.at(grad_lin, col, g)
                for b in range(len(batch_y)):
                    v_ij = embed[col[b]]
                    grad_emb[col[b]] += g[b] * (sum_v[b] - v_ij)

            grad_lin /= len(batch_y)
            grad_emb /= len(batch_y)

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias * grad_bias)
            bias -= lr * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
            v_lin = beta2 * v_lin + (1 - beta2) * (grad_lin * grad_lin)
            linear -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            embed -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # write weights.bin
    magic = 0x464D4C31  # 'FML1' little-endian
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias; float32 linear[d*256]; float32 embed[d*256*k]. */
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
    if (C != 2 || d <= 0 || k <= 0 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float logit = *bias;
        memset(sum_v, 0, sizeof(float) * k);
        memset(sum_v2, 0, sizeof(float) * k);

        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            logit += linear[idx];
            const float *e = embed + (long)idx * k;
            for (int f = 0; f < k; f++) {
                float v = e[f];
                sum_v[f] += v;
                sum_v2[f] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            interaction += sum_v[f] * sum_v[f] - sum_v2[f];
        }
        logit += 0.5f * interaction;

        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }

    free(sum_v);
    free(sum_v2);
    return 0;
}
```