```hypothesis
Family: label-fm (factorization machine with 4-dimensional embeddings). 
Why: FM captures pairwise feature interactions with O(d·k) parameters and compute, which suits the mixed ordinal/categorical Adult data. It should exceed label-linear accuracy (0.8524) while staying far cheaper than label-mlp16 (2325 Ir/item). 
Failure signal: dev accuracy ≤ 0.8524 or Ir/item ≥ 1500.
```

```python
"""Factorization Machine for tabular label tasks (k=4 embeddings).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weight format: magic 0x464D0001, int32 d, k, C; float32 linear[d][256], emb[d][256][k], bias."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 4
LR, L2, BATCH, EPOCHS = 0.01, 1e-4, 256, 10


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
    assert C == 2

    # split last 20% for validation (tuning split)
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # parameters
    lin = np.zeros((d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_b, v_b = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def predict(Xb):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        lin_sum = bias + np.sum(lin[np.arange(d), Xb.T].T, axis=1)  # (b,)
        emb_sum = np.zeros((b, K), np.float32)
        emb_sq_sum = np.zeros((b, K), np.float32)
        for j in range(d):
            e = emb[j, Xb[:, j]]  # (b, K)
            emb_sum += e
            emb_sq_sum += e * e
        inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
        return lin_sum + inter

    def loss_grad(Xb, yb):
        b = Xb.shape[0]
        logits = predict(Xb)
        # logistic loss gradient
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = p - yb  # (b,)
        # gradients
        g_bias = grad.mean()
        g_lin = np.zeros_like(lin)
        g_emb = np.zeros_like(emb)
        for j in range(d):
            vals = Xb[:, j]
            # linear gradient
            np.add.at(g_lin[j], vals, grad)
            # embedding gradient
            emb_sum_j = np.zeros((b, K), np.float32)
            for jj in range(d):
                emb_sum_j += emb[jj, Xb[:, jj]]
            # derivative of interaction term w.r.t emb[j, v] is (emb_sum_j - emb[j, v]) * grad
            for v in range(256):
                mask = (vals == v)
                if np.any(mask):
                    g_emb[j, v] = np.sum((emb_sum_j[mask] - emb[j, v]) * grad[mask, None], axis=0)
        g_lin /= b
        g_emb /= b
        g_lin += L2 * lin
        g_emb += L2 * emb
        return g_bias, g_lin, g_emb

    best_acc = -1.0
    best_params = None
    for epoch in range(EPOCHS):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx_b = order[s:s + BATCH]
            Xb, yb = X_tr[idx_b], y_tr[idx_b]
            g_b, g_lin, g_emb = loss_grad(Xb, yb)
            t += 1
            # Adam updates
            m_b = beta1 * m_b + (1 - beta1) * g_b
            v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
            bias -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            lin -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
        # validation
        logits_va = predict(X_va)
        preds_va = (logits_va > 0).astype(np.int64)
        acc = (preds_va == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = (lin.copy(), emb.copy(), bias)

    # retrain on all data with best hyperparameters (here we just keep the best from validation)
    lin, emb, bias = best_params

    # write weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += lin.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += np.array([bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0/1). */
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
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * d * 256 + 4L * d * 256 * K + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    float *lin = (float *)(wb + 16);
    float *emb = lin + d * 256;
    float bias = emb[d * 256 * K][0];
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)malloc(K * sizeof(float));
    float *emb_sq_sum = (float *)malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        for (int kk = 0; kk < K; kk++) { emb_sum[kk] = 0.0f; emb_sq_sum[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += lin[j * 256 + v];
            const float *e = emb + (j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                emb_sum[kk] += e[kk];
                emb_sq_sum[kk] += e[kk] * e[kk];
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += emb_sum[kk] * emb_sum[kk] - emb_sq_sum[kk];
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```