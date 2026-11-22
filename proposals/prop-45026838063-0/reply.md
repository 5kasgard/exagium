```hypothesis
Factorization Machine (FM) for tabular data: captures pairwise feature interactions via low-rank latent vectors.
Adult has 14 features (mix of ordinal/categorical) where interactions (e.g., education × occupation) are predictive.
An FM with latent dim k=8 adds ~28k parameters (vs ~3.5k for linear) and computes interactions in O(d·k) time.
We expect it to beat label-linear (0.8524) and approach label-mlp16 (0.8591) with lower inference cost than MLP.
If dev accuracy does not exceed 0.855, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: int32 magic 0x464D3130, d, k, C; float32 bias, linear[d][256], latent[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
K_CHOICES = [4, 8, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fm(X, y, d, k, rng, epochs=EPOCHS, batch=BATCH, lr=LR, l2=L2):
    n = X.shape[0]
    # params
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    latent = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / (256 + k))).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_lat, v_lat = np.zeros_like(latent), np.zeros_like(latent)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsize = len(idx)
            # forward
            lin_sum = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (b,)
            s_vec = np.sum(latent[np.arange(d), xb.T, :], axis=0)        # (b, k)
            sq_vec = np.sum(latent[np.arange(d), xb.T, :] ** 2, axis=0)  # (b, k)
            inter = 0.5 * np.sum(s_vec * s_vec - sq_vec, axis=1)         # (b,)
            logit = lin_sum + inter
            p = sigmoid(logit)
            grad_logit = (p - yb).astype(np.float32)                     # (b,)
            # gradients
            g_bias = np.sum(grad_logit)
            g_lin = np.zeros_like(linear)
            g_lat = np.zeros_like(latent)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad_logit)
                for l in range(k):
                    diff = s_vec[:, l] - latent[j, vals, l]
                    np.add.at(g_lat[j, :, l], vals, grad_logit * diff)
            # L2
            g_lin += 2 * l2 * linear
            g_lat += 2 * l2 * latent
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # latent
            m_lat = beta1 * m_lat + (1 - beta1) * g_lat
            v_lat = beta2 * v_lat + (1 - beta2) * g_lat * g_lat
            m_hat = m_lat / (1 - beta1 ** t)
            v_hat = v_lat / (1 - beta2 ** t)
            latent -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return bias, linear, latent

def evaluate(X, y, bias, linear, latent):
    n = X.shape[0]
    lin_sum = bias + np.sum(linear[np.arange(X.shape[1]), X.T], axis=0)
    s_vec = np.sum(latent[np.arange(X.shape[1]), X.T, :], axis=0)
    sq_vec = np.sum(latent[np.arange(X.shape[1]), X.T, :] ** 2, axis=0)
    inter = 0.5 * np.sum(s_vec * s_vec - sq_vec, axis=1)
    logit = lin_sum + inter
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    tune_size = int(0.2 * n)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_k, best_acc = 0, -1.0
    for k in K_CHOICES:
        bias, linear, latent = train_fm(X_train, y_train, d, k, rng)
        acc = evaluate(X_tune, y_tune, bias, linear, latent)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # retrain on all data
    bias, linear, latent = train_fm(X, y, d, best_k, rng)

    # write weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0/1). */
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
    if (C != 2 || k <= 0 || d <= 0 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *latent = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_vec = (float *)malloc(sizeof(float) * k);
    float *sq_vec = (float *)malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float lin_sum = bias[0];
        for (int l = 0; l < k; l++) { s_vec[l] = 0.0f; sq_vec[l] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += linear[(long)j * 256 + v];
            const float *lat = latent + ((long)j * 256 + v) * k;
            for (int l = 0; l < k; l++) {
                float val = lat[l];
                s_vec[l] += val;
                sq_vec[l] += val * val;
            }
        }

        float inter = 0.0f;
        for (int l = 0; l < k; l++) {
            inter += s_vec[l] * s_vec[l] - sq_vec[l];
        }
        inter *= 0.5f;

        float logit = lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(s_vec); free(sq_vec);
    return 0;
}
```