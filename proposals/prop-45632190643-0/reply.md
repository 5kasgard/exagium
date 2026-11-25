```hypothesis
Family: label-fm (factorization machine with shared embeddings). The adult dataset has 14 byte-valued features (binned numeric and categorical) where pairwise interactions are known to be predictive (e.g., education × occupation). A factorization machine models all pairwise interactions via low-rank embeddings (k=8) while keeping inference cost O(d·k) ≈ 112 multiply-adds per item, far below GBDT. We share the embedding vectors across the two classes and keep class-specific linear weights. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, k, C; float32 w0[C], w[d*256*C], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on last 20% tuning split, then trained on all)
SEED = 1
K = 8                # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50          # fixed after tuning
VAL_FRAC = 0.2       # last 20% for validation

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for validation (tuning split)
    split = int(n * (1 - VAL_FRAC))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    # Parameters
    w0 = np.zeros(C, np.float32)
    w = np.zeros((d, 256, C), np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    params = [w0, w, v]
    shapes = [p.shape for p in params]
    flat_shapes = [p.size for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    t = 0

    def forward(Xb):
        """Return logits (B, C) and cache for backward."""
        B = Xb.shape[0]
        # Linear part: w0 + sum_j w[j, x_j]
        logits = w0 + np.sum(w[np.arange(d), Xb.T].transpose(1, 0, 2), axis=1)  # (B, C)
        # FM part: 0.5 * ( (sum v)^2 - sum v^2 ) summed over embedding dim
        v_sum = np.sum(v[np.arange(d), Xb.T].transpose(1, 0, 2), axis=1)  # (B, K)
        v_sq_sum = np.sum(v[np.arange(d), Xb.T].transpose(1, 0, 2) ** 2, axis=1)  # (B, K)
        fm = 0.5 * np.sum(v_sum * v_sum - v_sq_sum, axis=1, keepdims=True)  # (B, 1)
        logits += fm
        return logits, (v_sum, v_sq_sum)

    def loss_grad(logits, yb):
        """Cross-entropy gradient w.r.t logits."""
        B = logits.shape[0]
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        return probs / B  # (B, C)

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, cache = forward(Xb)
            g_logits = loss_grad(logits, yb)  # (B, C)

            # Gradients
            gw0 = g_logits.sum(axis=0)  # (C,)
            gw = np.zeros_like(w)
            for j in range(d):
                vals = Xb[:, j]
                for c in range(C):
                    gw[j, vals, c] += g_logits[:, c]
            # FM gradients
            v_sum, v_sq_sum = cache
            # d fm / d v_sum = v_sum, d fm / d v_sq_sum = -0.5
            g_v_sum = v_sum * g_logits.sum(axis=1, keepdims=True)  # (B, K)
            g_v_sq_sum = -0.5 * g_logits.sum(axis=1, keepdims=True)  # (B, K)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = Xb[:, j]
                for k in range(K):
                    gv[j, vals, k] += g_v_sum[:, k] + 2 * v[j, vals, k] * g_v_sq_sum[:, k]

            # L2 regularization
            gw0 += L2 * w0
            gw += L2 * w
            gv += L2 * v

            grads = [gw0, gw, gv]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v_adam[i] = 0.999 * v_adam[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v_adam[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        val_logits, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]

    # Retrain on full data for best_epoch epochs
    w0, w, v = best_params
    params = [w0, w, v]
    m = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, cache = forward(Xb)
            g_logits = loss_grad(logits, yb)
            gw0 = g_logits.sum(axis=0)
            gw = np.zeros_like(w)
            for j in range(d):
                vals = Xb[:, j]
                for c in range(C):
                    gw[j, vals, c] += g_logits[:, c]
            v_sum, v_sq_sum = cache
            g_v_sum = v_sum * g_logits.sum(axis=1, keepdims=True)
            g_v_sq_sum = -0.5 * g_logits.sum(axis=1, keepdims=True)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = Xb[:, j]
                for k in range(K):
                    gv[j, vals, k] += g_v_sum[:, k] + 2 * v[j, vals, k] * g_v_sq_sum[:, k]
            gw0 += L2 * w0
            gw += L2 * w
            gv += L2 * v
            grads = [gw0, gw, gv]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v_adam[i] = 0.999 * v_adam[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v_adam[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<Iiii", magic, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index (0 or 1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w = w0 + C;
    const float *v = w + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)calloc(K, sizeof(float));
    float *sum_v_sq = (float *)calloc(K, sizeof(float));
    float logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        logits[0] = w0[0]; logits[1] = w0[1];
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            long base_w = ((long)j * 256 + val) * C;
            logits[0] += w[base_w];
            logits[1] += w[base_w + 1];
            long base_v = ((long)j * 256 + val) * K;
            for (int e = 0; e < K; e++) {
                float ve = v[base_v + e];
                sum_v[e] += ve;
                sum_v_sq[e] += ve * ve;
            }
        }
        float fm = 0.0f;
        for (int e = 0; e < K; e++) fm += sum_v[e] * sum_v[e] - sum_v_sq[e];
        fm *= 0.5f;
        logits[0] += fm; logits[1] += fm;

        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```