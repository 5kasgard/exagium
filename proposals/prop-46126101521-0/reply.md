```hypothesis
Factorization Machine (label-fm) with per-value embeddings captures pairwise feature interactions crucial for tabular data like Adult (e.g., education × occupation). Low-rank embeddings (K=4) keep inference fast: O(d·K) per item. If dev accuracy does not exceed label-linear (0.8524), the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameter grid for tuning split (last 20% of train.in)
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03]
EPOCHS_CHOICES = [5, 10, 20]
BATCH = 64
L2 = 1e-4
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fm(X, y, K, lr, epochs, rng):
    n, d = X.shape
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    e = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_e, v_e = np.zeros_like(e), np.zeros_like(e)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = len(yb)

            # gather embeddings
            # Xb: (bs, d) uint8
            # we need e_selected: (bs, d, K)
            e_sel = e[np.arange(d), Xb.T].transpose(1, 0, 2)  # (bs, d, K)
            w_sel = w[np.arange(d), Xb.T].transpose(1, 0)     # (bs, d)

            # linear part
            linear = w0 + w_sel.sum(axis=1)  # (bs,)

            # interaction part
            sum_e = e_sel.sum(axis=1)        # (bs, K)
            sum_e2 = (e_sel ** 2).sum(axis=1) # (bs, K)
            interaction = 0.5 * ((sum_e ** 2 - sum_e2).sum(axis=1))  # (bs,)

            logit = linear + interaction
            p = sigmoid(logit)
            g = (p - yb).astype(np.float32)  # (bs,)

            # gradients
            dw0 = g.mean()
            # dw[j, v] = mean over batch where Xb[:,j]==v of g
            dw = np.zeros_like(w)
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(dw[j], vals, g)
            dw /= bs

            # de: for each j, v, k: sum over i where Xb[i,j]==v of g[i] * (sum_e[i,k] - e_sel[i,j,k])
            de = np.zeros_like(e)
            for j in range(d):
                vals = Xb[:, j]
                # sum_e[i,k] - e_sel[i,j,k]
                diff = sum_e - e_sel[:, j, :]  # (bs, K)
                # multiply by g[i] -> (bs, K)
                grad = g[:, None] * diff
                np.add.at(de[j], vals, grad)
            de /= bs

            # L2 regularization on w and e (not w0)
            dw += L2 * w
            de += L2 * e

            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (dw0 * dw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw * dw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # e
            m_e = beta1 * m_e + (1 - beta1) * de
            v_e = beta2 * v_e + (1 - beta2) * (de * de)
            m_e_hat = m_e / (1 - beta1 ** t)
            v_e_hat = v_e / (1 - beta2 ** t)
            e -= lr * m_e_hat / (np.sqrt(v_e_hat) + eps)

    return w0, w, e

def eval_fm(X, y, w0, w, e):
    n, d = X.shape
    e_sel = e[np.arange(d), X.T].transpose(1, 0, 2)
    w_sel = w[np.arange(d), X.T].transpose(1, 0)
    linear = w0 + w_sel.sum(axis=1)
    sum_e = e_sel.sum(axis=1)
    sum_e2 = (e_sel ** 2).sum(axis=1)
    interaction = 0.5 * ((sum_e ** 2 - sum_e2).sum(axis=1))
    logit = linear + interaction
    pred = (logit > 0).astype(np.int64)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    n = X.shape[0]

    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_params = None
    best_model = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCHS_CHOICES:
                rng = np.random.default_rng(SEED)
                w0, w, e = train_fm(X_train, y_train, K, lr, epochs, rng)
                acc = eval_fm(X_val, y_val, w0, w, e)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, epochs)
                    best_model = (w0, w, e)
                print(f"Tune: K={K}, lr={lr}, epochs={epochs} -> val acc={acc:.4f}")

    print(f"Best: K={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]} -> {best_acc:.4f}")

    # retrain on full data with best params
    K, lr, epochs = best_params
    rng = np.random.default_rng(SEED)
    w0, w, e = train_fm(X, y, K, lr, epochs, rng)

    # write weights.bin
    # format: magic, d, K, C(=2), then w0, w[d][256], e[d][256][K] all float32 little-endian
    magic = 0x464D3100  # "FM1\0"
    C = 2
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0 || K > 256) { fprintf(stderr, "bad weights header\n"); return 3; }

    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *e = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        float sum_e[256]; // K max 256, but we know K<=8
        float sum_e2[256];
        for (int kk = 0; kk < K; kk++) { sum_e[kk] = 0.0f; sum_e2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *ejv = e + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = ejv[kk];
                sum_e[kk] += val;
                sum_e2[kk] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += 0.5f * (sum_e[kk] * sum_e[kk] - sum_e2[kk]);
        }
        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```