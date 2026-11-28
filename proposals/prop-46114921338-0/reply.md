```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult where combinations (e.g., education×occupation) are predictive. A rank-4 FM adds only ~14k parameters and ~364 multiply-adds per inference, cheaper than an MLP with 16 hidden units (~57k params). If dev accuracy does not exceed label-linear (0.8524), it suggests pairwise interactions are insufficient or the embedding rank is too small.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"K": 4, "lr": 0.01, "l2": 1e-4, "epochs": 10},
    {"K": 4, "lr": 0.003, "l2": 1e-4, "epochs": 10},
    {"K": 4, "lr": 0.01, "l2": 1e-3, "epochs": 10},
    {"K": 4, "lr": 0.003, "l2": 1e-3, "epochs": 10},
    {"K": 8, "lr": 0.01, "l2": 1e-4, "epochs": 10},
    {"K": 8, "lr": 0.003, "l2": 1e-4, "epochs": 10},
    {"K": 8, "lr": 0.01, "l2": 1e-3, "epochs": 10},
    {"K": 8, "lr": 0.003, "l2": 1e-3, "epochs": 10},
]
BATCH = 64
FINAL_EPOCHS = 20
MAGIC = 0x464D3101  # 'FM1' with version 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(idx, bias, linear, emb, K):
    """idx: (batch, d) uint8, returns logits (batch,)"""
    batch, d = idx.shape
    # linear part
    lin = bias + np.sum([linear[j, idx[:, j]] for j in range(d)], axis=0)
    # embeddings: (batch, d, K)
    e = np.stack([emb[j, idx[:, j], :] for j in range(d)], axis=1)
    sum_e = e.sum(axis=1)                 # (batch, K)
    sum_e_sq = (e**2).sum(axis=1)         # (batch, K)
    inter = 0.5 * np.sum(sum_e**2 - sum_e_sq, axis=1)
    return lin + inter

def train_fm(X, y, K, lr, l2, epochs, rng):
    n, d = X.shape
    # parameters
    bias = 0.0
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            batch = len(yb)
            # forward
            logits = fm_forward(xb, bias, linear, emb, K)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yb) / batch   # (batch,)
            # gradients
            g_bias = grad_logit.sum()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)
            # compute embeddings for this batch
            e = np.stack([emb[j, xb[:, j], :] for j in range(d)], axis=1)  # (batch, d, K)
            sum_e = e.sum(axis=1)                 # (batch, K)
            for j in range(d):
                vals = xb[:, j]
                # linear gradient
                np.add.at(g_lin[j], vals, grad_logit)
                # embedding gradient: grad_logit * (sum_e - e[:, j, :])
                g_emb_j = grad_logit[:, None] * (sum_e - e[:, j, :])  # (batch, K)
                np.add.at(g_emb[j], vals, g_emb_j)
            # L2 regularization
            g_lin += l2 * linear
            g_emb += l2 * emb
            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1**t)
            v_hat = v_bias / (1 - beta2**t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= lr * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= lr * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)
    return bias, linear, emb

def evaluate(X, y, bias, linear, emb, K):
    logits = fm_forward(X, bias, linear, emb, K)
    pred = (logits > 0).astype(np.int64)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)
    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    # grid search
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for cfg in GRID:
        bias, linear, emb = train_fm(X_train, y_train, cfg["K"], cfg["lr"], cfg["l2"], cfg["epochs"], rng)
        acc = evaluate(X_tune, y_tune, bias, linear, emb, cfg["K"])
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (bias, linear, emb)
    # retrain on full data with best config
    print(f"Best config: {best_cfg}, tune acc: {best_acc:.4f}", file=sys.stderr)
    bias, linear, emb = train_fm(X, y, best_cfg["K"], best_cfg["lr"], best_cfg["l2"], FINAL_EPOCHS, rng)
    # save weights
    K = best_cfg["K"]
    out = struct.pack("<4i", MAGIC, d, K, 1)  # C=1 for binary logit
    out += struct.pack("<f", bias)
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3101) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = *p++;
    const float *linear = p; p += (long)d * 256;
    const float *emb = p;  // size d * 256 * K
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);
    float *emb_j = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        // linear part
        for (int j = 0; j < d; j++) logit += linear[j * 256 + xi[j]];
        // interaction part: 0.5 * ( (sum emb)^2 - sum(emb^2) )
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_emb_sq, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float v = e[u];
                sum_emb[u] += v;
                sum_emb_sq[u] += v * v;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = sum_emb[u];
            inter += 0.5f * (s * s - sum_emb_sq[u]);
        }
        logit += inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```