```hypothesis
Family: label-fm (Factorization Machine with per-value embeddings).
Reason: The Adult dataset has 14 categorical/ordinal features each with 256 values. A linear model (label-linear) ignores feature interactions. A Factorization Machine adds pairwise interactions via low-rank embeddings (size k=8), capturing non-linear effects while keeping inference at O(d*k) ~ 112 FLOPs/item. It should beat label-linear's 0.8524 accuracy and approach label-gbdt's 0.8664 at a fraction of the CPU cost (target < 1000 Ir/item). Failure would be dev accuracy ≤ 0.853 or Ir/item > 1500.
```
```python
"""Factorization Machine for tabular label tasks with per-byte lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, k, C=1; float32 w0, linear[d][256], fm[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 50
MAX_EPOCHS_FULL = 100
PATIENCE = 5
GRID = {
    'k': [4, 8, 16],
    'lr': [1e-2, 3e-3, 1e-3],
    'l2': [1e-4, 1e-5],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward(X, w0, linear, fm, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    lin = w0 + np.sum(linear[np.arange(d), X.T], axis=0)  # (n,)
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_emb_sq = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        emb = fm[j, X[:, j]]  # (n, k)
        sum_emb += emb
        sum_emb_sq += emb * emb
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    return lin + inter

def accuracy(X, y, w0, linear, fm, k):
    logits = forward(X, w0, linear, fm, k)
    preds = (sigmoid(logits) > 0.5).astype(np.int64)
    return np.mean(preds == y)

def train_epoch(X, y, w0, linear, fm, k, lr, l2, rng, opt_state):
    """One epoch over X,y with Adam. Returns updated params and opt_state."""
    n, d = X.shape
    order = rng.permutation(n)
    t = opt_state['t']
    m_w0, v_w0 = opt_state['m_w0'], opt_state['v_w0']
    m_lin, v_lin = opt_state['m_lin'], opt_state['v_lin']
    m_fm, v_fm = opt_state['m_fm'], opt_state['v_fm']
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb, yb = X[idx], y[idx]
        bs = len(yb)
        # forward
        lin_part = w0 + np.sum(linear[np.arange(d), xb.T], axis=0)
        sum_emb = np.zeros((bs, k), np.float32)
        sum_emb_sq = np.zeros((bs, k), np.float32)
        embs = []
        for j in range(d):
            e = fm[j, xb[:, j]]
            embs.append(e)
            sum_emb += e
            sum_emb_sq += e * e
        inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
        logits = lin_part + inter
        probs = sigmoid(logits)
        # gradients
        dlogit = (probs - yb) / bs  # (bs,)
        # w0
        gw0 = np.sum(dlogit) + l2 * w0
        # linear
        glin = np.zeros_like(linear)
        for j in range(d):
            np.add.at(glin[j], xb[:, j], dlogit)
        glin += l2 * linear
        # fm
        gfm = np.zeros_like(fm)
        for j in range(d):
            e = embs[j]
            grad_e = dlogit[:, None] * (sum_emb - e)  # (bs, k)
            for f in range(k):
                np.add.at(gfm[j, :, f], xb[:, j], grad_e[:, f])
        gfm += l2 * fm
        # Adam updates
        t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        # w0
        m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
        v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
        m_hat = m_w0 / (1 - beta1 ** t)
        v_hat = v_w0 / (1 - beta2 ** t)
        w0 -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # linear
        m_lin = beta1 * m_lin + (1 - beta1) * glin
        v_lin = beta2 * v_lin + (1 - beta2) * glin * glin
        m_hat = m_lin / (1 - beta1 ** t)
        v_hat = v_lin / (1 - beta2 ** t)
        linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # fm
        m_fm = beta1 * m_fm + (1 - beta1) * gfm
        v_fm = beta2 * v_fm + (1 - beta2) * gfm * gfm
        m_hat = m_fm / (1 - beta1 ** t)
        v_hat = v_fm / (1 - beta2 ** t)
        fm -= lr * m_hat / (np.sqrt(v_hat) + eps)
    opt_state.update(t=t, m_w0=m_w0, v_w0=v_w0, m_lin=m_lin, v_lin=v_lin, m_fm=m_fm, v_fm=v_fm)
    return w0, linear, fm

def init_params(d, k, rng):
    w0 = np.float32(0.0)
    linear = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    fm = rng.uniform(-0.01, 0.01, (d, 256, k)).astype(np.float32)
    return w0, linear, fm

def init_opt_state(d, k):
    return {
        't': 0,
        'm_w0': np.float32(0.0), 'v_w0': np.float32(0.0),
        'm_lin': np.zeros((d, 256), np.float32), 'v_lin': np.zeros((d, 256), np.float32),
        'm_fm': np.zeros((d, 256, k), np.float32), 'v_fm': np.zeros((d, 256, k), np.float32),
    }

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)
    # tuning split: last 20%
    n = len(X)
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_acc = -1.0
    best_hp = None
    best_params = None
    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                w0, linear, fm = init_params(d, k, rng)
                opt = init_opt_state(d, k)
                best_va_acc = -1.0
                patience = 0
                for epoch in range(MAX_EPOCHS_TUNE):
                    w0, linear, fm = train_epoch(X_tr, y_tr, w0, linear, fm, k, lr, l2, rng, opt)
                    va_acc = accuracy(X_va, y_va, w0, linear, fm, k)
                    if va_acc > best_va_acc:
                        best_va_acc = va_acc
                        patience = 0
                    else:
                        patience += 1
                        if patience >= PATIENCE:
                            break
                if best_va_acc > best_acc:
                    best_acc = best_va_acc
                    best_hp = (k, lr, l2)
                    best_params = (w0.copy(), linear.copy(), fm.copy())
    print(f"Best hp: k={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data
    k, lr, l2 = best_hp
    w0, linear, fm = init_params(d, k, rng)
    opt = init_opt_state(d, k)
    for epoch in range(MAX_EPOCHS_FULL):
        w0, linear, fm = train_epoch(X, y, w0, linear, fm, k, lr, l2, rng, opt)
    # export
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, k, 1)
    out += np.array([w0], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += fm.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Weights format: int32 magic(0x464D4C31), d, k, C=1; float32 w0, linear[d][256], fm[d][256][k].
 * Input record: d bytes. Output record: 1 byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    long expected = 16L + 4L * (1L + (long)d * 256L + (long)d * 256L * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expect %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *linear = p + 1;
    const float *fm = linear + (long)d * 256L;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) logit += linear[(long)j * 256 + xi[j]];
        for (int f = 0; f < k; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *emb = fm + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float v = emb[f];
                sum_emb[f] += v;
                sum_emb_sq[f] += v * v;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
        logit += 0.5f * inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char cls = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```