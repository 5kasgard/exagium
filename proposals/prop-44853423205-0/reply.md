```hypothesis
Factorization Machine (label-fm) captures pairwise feature interactions via low-rank embeddings (k=8), improving on linear models for this tabular dataset with 14 categorical fields. Inference adds only 91*k multiply-adds per item (k=8 => ~728 ops), keeping CPU cost near label-linear. If dev accuracy <= 0.8524 or instructions/item > 2325, the model fails to beat label-linear or label-mlp16.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning on last 20% of train
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03, 0.1]
L2_CHOICES = [1e-4, 1e-3, 1e-2]
EPOCHS = 30
BATCH = 256
VAL_FRACTION = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, global_bias, linear_w, emb, k):
    """X: (n, d) uint8, returns scores (n,)"""
    n, d = X.shape
    # linear term
    lin = global_bias + np.sum(linear_w[np.arange(d), X.T], axis=0)
    # interaction term via FM trick
    # emb shape (d, 256, k)
    # get embeddings for each feature: (d, n, k)
    emb_vals = emb[np.arange(d), X.T]  # (d, n, k)
    sum_emb = np.sum(emb_vals, axis=0)  # (n, k)
    sum_emb_sq = np.sum(emb_vals ** 2, axis=0)
    inter = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)
    return lin + inter


def train_fm(X_train, y_train, X_val, y_val, k, lr, l2, epochs, batch, rng):
    n, d = X_train.shape
    # initialize parameters
    global_bias = np.float32(0.0)
    linear_w = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)

    # Adam state
    m_gb, v_gb = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear_w), np.zeros_like(linear_w)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X_train[idx]
            yb = y_train[idx]
            bsz = len(yb)

            # forward
            scores = fm_forward(xb, global_bias, linear_w, emb, k)
            probs = sigmoid(scores)
            # gradients
            grad = (probs - yb).astype(np.float32)  # (bsz,)
            # global bias
            g_gb = grad.mean()
            # linear weights
            g_lin = np.zeros_like(linear_w)
            for i in range(d):
                vals = xb[:, i]
                np.add.at(g_lin[i], vals, grad)
            g_lin /= bsz
            g_lin += l2 * linear_w
            # embeddings
            g_emb = np.zeros_like(emb)
            # compute sum_emb for batch
            emb_vals = emb[np.arange(d), xb.T]  # (d, bsz, k)
            sum_emb = np.sum(emb_vals, axis=0)  # (bsz, k)
            for i in range(d):
                vals = xb[:, i]
                e_i = emb[i, vals]  # (bsz, k)
                # gradient for feature i: (sum_emb - e_i) * grad[:, None]
                g_emb[i, vals] += (grad[:, None] * (sum_emb - e_i)).astype(np.float32)
            g_emb /= bsz
            g_emb += l2 * emb

            # Adam update
            t += 1
            # global bias
            m_gb = beta1 * m_gb + (1 - beta1) * g_gb
            v_gb = beta2 * v_gb + (1 - beta2) * g_gb * g_gb
            m_gb_hat = m_gb / (1 - beta1 ** t)
            v_gb_hat = v_gb / (1 - beta2 ** t)
            global_bias -= lr * m_gb_hat / (np.sqrt(v_gb_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            linear_w -= lr * m_lin_hat / (np.sqrt(v_lin_hat) + eps)

            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

        # validation accuracy
        val_scores = fm_forward(X_val, global_bias, linear_w, emb, k)
        val_pred = (val_scores > 0).astype(np.int64)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (global_bias.copy(), linear_w.copy(), emb.copy())

    return best_params, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # split: last 20% for validation
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)

    best_overall = None
    best_val = -1.0
    best_hp = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # set seed for each run to be deterministic
                run_rng = np.random.default_rng(SEED)
                params, val_acc = train_fm(X_train, y_train, X_val, y_val, k, lr, l2, EPOCHS, BATCH, run_rng)
                if val_acc > best_val:
                    best_val = val_acc
                    best_overall = params
                    best_hp = (k, lr, l2)

    print(f"Best hyperparams: k={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, val_acc={best_val:.4f}", file=sys.stderr)

    # retrain on full data with best hyperparams
    k, lr, l2 = best_hp
    full_rng = np.random.default_rng(SEED)
    final_params, _ = train_fm(X, y, X_val, y_val, k, lr, l2, EPOCHS, BATCH, full_rng)
    global_bias, linear_w, emb = final_params

    # write weights.bin
    # format: magic(4), d(4), k(4), C(4), global_bias(4), linear_w[d*256](4), emb[d*256*k](4)
    magic = 0x464D3130  # "FM10"
    C = card["classes"]
    out = struct.pack("<5i", magic, d, k, C, 0)  # placeholder for global_bias? we'll pack separately
    # Actually pack as: magic, d, k, C, then float32 global_bias, then linear_w, then emb
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", global_bias)
    out += linear_w.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float global_bias = p[0];
    const float *linear_w = p + 1;
    const float *emb = linear_w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }

        float score = global_bias;
        // linear term
        for (int j = 0; j < d; j++) {
            score += linear_w[j * 256 + xi[j]];
        }
        // interaction term via FM trick
        // accumulate sum of embeddings per factor
        float sum_emb[32] = {0}; // k <= 32
        float sum_emb_sq[32] = {0};
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float v = e[f];
                sum_emb[f] += v;
                sum_emb_sq[f] += v * v;
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```