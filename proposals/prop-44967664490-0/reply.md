```hypothesis
Family: label-fm (Factorization Machine for tabular data).
Why: The Adult dataset has known feature interactions (e.g., education–occupation, capital-gain–hours-per-week). A degree-2 FM captures all pairwise interactions with O(d·k) parameters, which is expressive yet compact. With k=8 it adds only ~28k parameters over linear, and inference is a few hundred FLOPs per item. It should beat linear (0.8524) and approach MLP16 (0.8591) at lower cost.
Failure: If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear by >2×.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Encodes all 14 features as categorical (256 values each). Learns per-feature-value
linear weights and k-dimensional embeddings. Trains with Adam on logistic loss.
Exports: magic 0x464D4C31, d, k, C; float32 linear[d*256], emb[d*256*k], bias."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8                     # embedding dimension
LR = 1e-2
L2 = 1e-5
BATCH = 256
MAX_EPOCHS = 50
PATIENCE = 5              # early stopping on validation split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]          # 14
    C = card["classes"]                 # 2
    n = X.shape[0]

    # split: last 20% for validation (tuning split), first 80% for training
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # parameters
    W_lin = np.zeros((d, 256), dtype=np.float32)
    W_emb = np.zeros((d, 256, K), dtype=np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_lin = np.zeros_like(W_lin); v_lin = np.zeros_like(W_lin)
    m_emb = np.zeros_like(W_emb); v_emb = np.zeros_like(W_emb)
    m_bias = np.float32(0.0); v_bias = np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(Xb):
        # linear term: sum_j W_lin[j, x_j]
        lin = bias + np.sum(W_lin[np.arange(d), Xb.T], axis=0)
        # FM term: 0.5 * sum_f ( (sum_j emb[j, x_j, f])^2 - sum_j emb[j, x_j, f]^2 )
        emb_vals = W_emb[np.arange(d), Xb.T]          # (d, batch, K)
        sum_emb = np.sum(emb_vals, axis=0)            # (batch, K)
        sum_emb_sq = np.sum(emb_vals * emb_vals, axis=0)
        fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
        return lin + fm

    def loss_grad(Xb, yb):
        logits = forward(Xb)
        p = sigmoid(logits)
        grad = (p - yb).astype(np.float32)            # (batch,)
        # linear gradients
        g_lin = np.zeros_like(W_lin)
        np.add.at(g_lin, (np.arange(d)[:, None], Xb.T), grad)
        g_lin /= len(yb)
        g_lin += L2 * W_lin
        # embedding gradients
        g_emb = np.zeros_like(W_emb)
        emb_vals = W_emb[np.arange(d), Xb.T]          # (d, batch, K)
        sum_emb = np.sum(emb_vals, axis=0)            # (batch, K)
        # d/dv_{j,f} = grad * (sum_emb[:,f] - emb_vals[j,:,f])
        for j in range(d):
            xj = Xb[:, j]
            for f in range(K):
                g_emb[j, xj, f] += grad * (sum_emb[:, f] - emb_vals[j, np.arange(len(yb)), f])
        g_emb /= len(yb)
        g_emb += L2 * W_emb
        # bias gradient
        g_bias = np.float32(np.mean(grad))
        return g_lin, g_emb, g_bias

    best_val_loss = float('inf')
    best_epoch = 0
    best_params = None

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            g_lin, g_emb, g_bias = loss_grad(Xb, yb)
            t += 1
            # Adam update
            for param, grad, m, v in [(W_lin, g_lin, m_lin, v_lin),
                                      (W_emb, g_emb, m_emb, v_emb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat_b = m_bias / (1 - beta1 ** t)
            v_hat_b = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # validation
        val_logits = forward(X_va)
        val_loss = np.mean(np.log(1 + np.exp(-val_logits * (2 * y_va - 1))))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = (W_lin.copy(), W_emb.copy(), bias)
        elif epoch - best_epoch >= PATIENCE:
            break

    # retrain on full data for best_epoch+1 epochs
    W_lin, W_emb, bias = best_params
    # reset Adam state
    m_lin.fill(0); v_lin.fill(0)
    m_emb.fill(0); v_emb.fill(0)
    m_bias = 0.0; v_bias = 0.0
    t = 0
    for epoch in range(best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            g_lin, g_emb, g_bias = loss_grad(Xb, yb)
            t += 1
            for param, grad, m, v in [(W_lin, g_lin, m_lin, v_lin),
                                      (W_emb, g_emb, m_emb, v_emb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat_b = m_bias / (1 - beta1 ** t)
            v_hat_b = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # write weights.bin
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += W_emb.astype("<f4").tobytes()
    out += np.array([bias], dtype="<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t lin_sz = (size_t)d * 256;
    size_t emb_sz = (size_t)d * 256 * K;
    long expected = 16 + 4L * (lin_sz + emb_sz + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *emb = lin + lin_sz;
    const float *bias = emb + emb_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_emb_sq, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += lin[j * 256 + v];
            const float *e = emb + (size_t)j * 256 * K + v * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```