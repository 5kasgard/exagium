```hypothesis
Family: label-fm (Factorization Machine for tabular data).
Why: Adult has 14 mixed-type features with many categorical values; FM captures pairwise feature interactions via low-rank embeddings, which linear models miss. With embedding dim k=8, inference is O(d*k) ~ 112 multiply-adds per item, far cheaper than GBDT/MLP while often matching their accuracy on tabular tasks.
Failure signal: If dev accuracy <= 0.8524 (label-linear) or CPU cost >= label-mlp16, the model does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Trains a binary FM with logistic loss using Adam. Hyperparameters chosen on last 20% of train.in (tuning split),
then retrained on full train. Exports: magic 0x464D3132, d, k, C, float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 256
MAX_EPOCHS = 20
L2 = 1e-4
LR = 0.03
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
EPOCH_CHOICES = [10, 20]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, w0, w, V, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    # linear part
    lin = w0 + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part
    inter = np.zeros(n, dtype=np.float32)
    for f in range(k):
        Vf = V[:, :, f]  # (d, 256)
        v_sum = np.sum(Vf[np.arange(d), X.T], axis=0)  # (n,)
        v_sq_sum = np.sum(Vf[np.arange(d), X.T] ** 2, axis=0)  # (n,)
        inter += 0.5 * (v_sum * v_sum - v_sq_sum)
    return lin + inter


def train_fm(X, y, k, lr, epochs, rng):
    n, d = X.shape
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = np.zeros((d, 256, k), dtype=np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            bsz = len(yi)

            # forward
            logits = fm_forward(xi, w0, w, V, k)
            p = sigmoid(logits)
            loss_grad = (p - yi).astype(np.float32)  # (bsz,)

            # gradients
            # linear: w0, w
            grad_w0 = np.mean(loss_grad)
            grad_w = np.zeros_like(w)
            for j in range(d):
                vals = xi[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        grad_w[j, v] = np.mean(loss_grad[mask])
            grad_w += L2 * w

            # V gradients
            grad_V = np.zeros_like(V)
            for f in range(k):
                Vf = V[:, :, f]
                v_sum = np.sum(Vf[np.arange(d), xi.T], axis=0)  # (bsz,)
                for j in range(d):
                    vals = xi[:, j]
                    v_j = Vf[j, vals]  # (bsz,)
                    # d/dV_j = loss_grad * (v_sum - v_j)
                    g = loss_grad * (v_sum - v_j)
                    for v in range(256):
                        mask = (vals == v)
                        if np.any(mask):
                            grad_V[j, v, f] = np.mean(g[mask])
            grad_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    return w0, w, V


def eval_acc(X, y, w0, w, V, k):
    logits = fm_forward(X, w0, w, V, k)
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    n = len(y_all)
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCH_CHOICES:
                w0, w, V = train_fm(X_train, y_train, k, lr, epochs, rng)
                acc = eval_acc(X_val, y_val, w0, w, V, k)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, epochs)
                    best_model = (w0, w, V)

    k, lr, epochs = best_params
    # retrain on full data
    rng = np.random.default_rng(SEED)
    w0, w, V = train_fm(X_all, y_all, k, lr, epochs, rng)

    # write weights
    magic = 0x464D3132  # 'F','M','1','2'
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3132) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float logit = w0;
        // linear part
        for (int j = 0; j < d; j++) logit += w[j * 256 + xi[j]];

        // interaction part
        for (int f = 0; f < k; f++) {
            float sum = 0.0f, sq_sum = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[(j * 256 + xi[j]) * k + f];
                sum += v;
                sq_sum += v * v;
            }
            logit += 0.5f * (sum * sum - sq_sum);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```