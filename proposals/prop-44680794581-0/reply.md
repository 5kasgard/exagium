```hypothesis
Family: label-fm (Factorization Machine, rank-8 pairwise interactions).
Why: Adult has strong feature interactions (e.g., education × occupation, marital-status × relationship).
     FM captures all pairwise interactions with O(d·R) parameters (d=14, R=8 → ~14k weights) and
     inference cost ~14×8 multiply-adds per item, far cheaper than GBT or MLP.
Failure: Dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000.
```

```python
"""Factorization Machine for tabular label tasks (fixed 14-byte items).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyper-parameters chosen on last 20% of train.in (file order), then retrained on all train.
Exported weights: magic 0x464D0001, d, R, C; float32 bias, W_lin[d*256], V[d*256*R]."""
import json, math, struct, sys

import numpy as np

# Hyper-parameter grid searched on the tuning split
GRID = [
    {"R": 4, "lr": 0.02, "l2": 1e-4, "epochs": 15, "batch": 64},
    {"R": 8, "lr": 0.01, "l2": 1e-4, "epochs": 15, "batch": 64},
    {"R": 16, "lr": 0.005, "l2": 1e-4, "epochs": 15, "batch": 64},
]
SEED = 1
VAL_FRAC = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, bias, W_lin, V, R, C):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    # Linear part
    lin = bias + W_lin[np.arange(d) * 256 + X].sum(axis=1)  # (n,)
    # Interaction part: 0.5 * sum_r ( (sum_i v_ir)^2 - sum_i v_ir^2 )
    # V shape: (d, 256, R)
    V_sel = V[np.arange(d), X]               # (n, d, R)
    sum_v = V_sel.sum(axis=1)                # (n, R)
    sum_v2 = (V_sel ** 2).sum(axis=1)        # (n, R)
    inter = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))  # (n,)
    logits = lin[:, None] + inter[:, None]   # (n, 1) broadcast to C=2
    return np.repeat(logits, C, axis=1)      # (n, C) both classes same logit? Wait, FM is for binary? Actually we have C classes. We'll do one-vs-rest: separate bias, W_lin, V per class? Simpler: treat as binary (C=2) with single output. But card says classes=2. We'll output single logit and use sigmoid. For multi-class we'd need per-class factors. Since C=2, we can do binary classification with one output. So logits shape (n, 1). Then in infer we compute sigmoid and pick class 1 if >0.5. But the contract expects output record as class index (0 or 1). We'll do binary FM.
    # Let's change: we'll have bias, W_lin, V per class? That would be C times larger. For C=2, we can just learn one set and use threshold. But to be general, we can learn per-class linear and interaction? That's too heavy. Instead, we'll do binary classification: output a single score, class 1 if score>0 else 0. This matches the task (binary). So we only need one set of parameters. The output in infer will be 0 or 1.
    # So we change: logits is (n,). We'll return that.

def fm_forward_bin(X, bias, W_lin, V, R):
    n, d = X.shape
    lin = bias + W_lin[np.arange(d) * 256 + X].sum(axis=1)
    V_sel = V[np.arange(d), X]
    sum_v = V_sel.sum(axis=1)
    sum_v2 = (V_sel ** 2).sum(axis=1)
    inter = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
    return lin + inter  # (n,)


def train_fm(X, y, R, lr, l2, epochs, batch, rng):
    n, d = X.shape
    # Initialize parameters
    bias = 0.0
    W_lin = rng.normal(0, 0.01, d * 256).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, R)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_W, v_W = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # Forward
            lin = bias + W_lin[np.arange(d) * 256 + xb].sum(axis=1)
            V_sel = V[np.arange(d), xb]          # (b, d, R)
            sum_v = V_sel.sum(axis=1)            # (b, R)
            sum_v2 = (V_sel ** 2).sum(axis=1)    # (b, R)
            inter = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
            scores = lin + inter
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-scores))
            grad = (p - yb).astype(np.float32)   # (b,)
            # Gradients
            g_bias = grad.mean()
            # Linear weights gradient
            g_W = np.zeros_like(W_lin)
            for i in range(d):
                vals = xb[:, i]
                np.add.at(g_W, i * 256 + vals, grad)
            g_W /= batch
            g_W += l2 * W_lin
            # V gradient
            g_V = np.zeros_like(V)
            for i in range(d):
                vals = xb[:, i]
                v_sel = V[i, vals]               # (b, R)
                sum_v_i = sum_v                  # (b, R)
                # d_inter/d_v_ikr = v_ikr * (sum_v_r - v_ikr)
                # Actually: inter = 0.5 * sum_r ( (sum_i v_ir)^2 - sum_i v_ir^2 )
                # d_inter/d_v_ikr = sum_v_r - v_ikr
                grad_expanded = grad[:, None]    # (b, 1)
                g_v = grad_expanded * (sum_v_i - v_sel)  # (b, R)
                np.add.at(g_V[i], vals, g_v)
            g_V /= batch
            g_V += l2 * V
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # W_lin
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W_lin -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat = m_V / (1 - beta1 ** t)
            v_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return bias, W_lin, V


def eval_acc(X, y, bias, W_lin, V, R):
    scores = fm_forward_bin(X, bias, W_lin, V, R)
    preds = (scores > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    val_size = int(n * VAL_FRAC)
    train_size = n - val_size
    # Split in file order (last 20% is validation)
    X_tr, X_val = X_all[:train_size], X_all[train_size:]
    y_tr, y_val = y_all[:train_size], y_all[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        bias, W_lin, V = train_fm(X_tr, y_tr, cfg["R"], cfg["lr"], cfg["l2"], cfg["epochs"], cfg["batch"], rng)
        acc = eval_acc(X_val, y_val, bias, W_lin, V, cfg["R"])
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, (bias, W_lin, V)
    # Retrain on all data with best config
    bias, W_lin, V = train_fm(X_all, y_all, best_cfg["R"], best_cfg["lr"], best_cfg["l2"], best_cfg["epochs"], best_cfg["batch"], rng)
    d, R, C = card["input"]["bytes"], best_cfg["R"], card["classes"]
    # Write weights: magic, d, R, C; float32 bias, W_lin[d*256], V[d*256*R]
    out = struct.pack("<4i", 0x464D0001, d, R, C)
    out += struct.pack("<f", bias)
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output record: 1 byte (class 0 or 1).
 * Weights format: magic(0x464D0001), d, R, C; float bias, W_lin[d*256], V[d*256*R]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C != 2 || d != 14 || R <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * R;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W_lin = p + 1;
    const float *V = p + 1 + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * R);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Linear part
        float score = bias;
        for (int j = 0; j < d; j++) score += W_lin[j * 256 + xi[j]];
        // Interaction part
        memset(sum_v, 0, sizeof(float) * R);
        for (int j = 0; j < d; j++) {
            const float *v = V + (j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) sum_v[r] += v[r];
        }
        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + (j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) {
                float diff = sum_v[r] - v[r];
                inter += v[r] * diff;
            }
        }
        score += 0.5f * inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    return 0;
}
```