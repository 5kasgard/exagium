```hypothesis
Family: label-fm (factorization machine). FMs model pairwise interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features. With embedding dim 8, the model has ~32k parameters, offering higher expressivity than linear models at O(d*k) inference cost. We expect it to beat label-linear (0.8524) and approach label-gbdt (0.8664) with lower CPU cost. Failure if dev accuracy < 0.85.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split)
K = 8                # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 5

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
    n, d = X.shape
    C = card["classes"]
    assert C == 2, "FM binary only"

    # Split: last 20% for validation (tuning split)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    bias = np.array([math.log((y_tr == 1).mean() / (y_tr == 0).mean())], dtype=np.float32)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    def forward(Xb):
        # Xb: (batch, d)
        batch = Xb.shape[0]
        lin_sum = np.sum(linear[np.arange(d), Xb.T].T, axis=1)  # (batch,)
        # embeddings: (batch, d, K)
        emb_batch = emb[np.arange(d), Xb.T].transpose(1, 0, 2)
        emb_sum = np.sum(emb_batch, axis=1)          # (batch, K)
        emb_sq_sum = np.sum(emb_batch ** 2, axis=1)  # (batch, K)
        inter = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1)
        logit = bias[0] + lin_sum + inter
        return logit, lin_sum, emb_sum, emb_batch

    def loss_grad(logit, yb):
        # BCE with logits gradient
        p = 1.0 / (1.0 + np.exp(-logit))
        grad = (p - yb).astype(np.float32)  # (batch,)
        return grad

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logit, lin_sum, emb_sum, emb_batch = forward(Xb)
            grad = loss_grad(logit, yb)[:, None]  # (batch, 1)

            # Gradients
            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            # Linear gradients: for each feature j, value v, add grad
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(g_lin[j], vals, grad[:, 0])

            # Embedding gradients: g_emb[j, v] += grad * (emb_sum - emb_batch[:, j])
            for j in range(d):
                vals = Xb[:, j]
                diff = emb_sum - emb_batch[:, j]  # (batch, K)
                for k in range(K):
                    np.add.at(g_emb[j, :, k], vals, grad[:, 0] * diff[:, k])

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            m_hat = m_bias / (1 - 0.9 ** t)
            v_hat = v_bias / (1 - 0.999 ** t)
            bias[0] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)

            # emb
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

        # Validation
        logit_val, _, _, _ = forward(X_val)
        pred_val = (logit_val > 0).astype(np.int64)
        val_acc = (pred_val == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = (bias.copy(), linear.copy(), emb.copy())
        elif epoch - best_epoch >= PATIENCE:
            break

    # Retrain on full data for best_epoch+1 epochs
    bias, linear, emb = best_params
    # Reset Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    for epoch in range(best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logit, lin_sum, emb_sum, emb_batch = forward(Xb)
            grad = loss_grad(logit, yb)[:, None]

            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for j in range(d):
                vals = Xb[:, j]
                np.add.at(g_lin[j], vals, grad[:, 0])

            for j in range(d):
                vals = Xb[:, j]
                diff = emb_sum - emb_batch[:, j]
                for k in range(K):
                    np.add.at(g_emb[j, :, k], vals, grad[:, 0] * diff[:, k])

            g_lin += L2 * linear
            g_emb += L2 * emb

            t += 1
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            m_hat = m_bias / (1 - 0.9 ** t)
            v_hat = v_bias / (1 - 0.999 ** t)
            bias[0] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)

            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    # Format: magic(4), d(4), K(4), C(4), bias(4), linear[d*256*4], emb[d*256*K*4]
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = 0.0f;
        for (int j = 0; j < K; j++) { emb_sum[j] = 0.0f; emb_sq_sum[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float ev = e[kk];
                emb_sum[kk] += ev;
                emb_sq_sum[kk] += ev * ev;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = emb_sum[kk];
            inter += s * s - emb_sq_sum[kk];
        }
        inter *= 0.5f;

        float logit = bias + lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(emb_sum); free(emb_sq_sum);
    if (x) free(x);
    free(wb);
    return 0;
}
```