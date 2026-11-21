```hypothesis
Family: label-fm (Factorization Machine with shared interactions).  
Why: Adult has mixed ordinal/categorical features; FMs capture pairwise interactions efficiently via low-rank factors, often beating linear models on tabular data. Shared interaction term keeps parameters and inference cost low.  
Failure signal: If dev accuracy ≤ label-linear (0.8524) or instructions/item exceed label-mlp16 (2325), the factor dimension or training regime is insufficient.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored scalar, categorical as one-hot over training values.
Trains FM with linear term + shared pairwise interactions (factor dim F).
Chooses F on last 20% of train.in (tuning split), then retrains on all data.
Exports per-byte lookup tables: magic 0x464D4C31, d, F, C; float32 linear_T[d][256][C], bias_lin[C], factor_T[d][256][F]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_F = [4, 8, 16]  # factor dimensions to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(card, X):
    """Return list of encoding matrices E_j (256 x k_j) and total k."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode_batch(X, enc):
    """X: (n, d) uint8, enc: list of (256, k_j) -> Z: (n, k) float32"""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def fm_forward(Z, w, v, b):
    """Z: (n, k), w: (k, C), v: (k, F), b: (C) -> logits (n, C)"""
    n, k = Z.shape
    _, C = w.shape
    _, F = v.shape
    lin = Z @ w + b  # (n, C)
    # interaction term (shared across classes)
    # s_f = Z @ v[:, f]  (n,)
    # s2_f = (Z^2) @ (v[:, f]^2)
    Z2 = Z * Z
    v2 = v * v
    s = Z @ v          # (n, F)
    s2 = Z2 @ v2       # (n, F)
    inter = 0.5 * np.sum(s * s - s2, axis=1, keepdims=True)  # (n, 1)
    return lin + inter  # broadcast inter to (n, C)

def fm_backward(Z, y, w, v, b, probs):
    """Compute gradients for w, v, b. probs: (n, C) softmax probs."""
    n, k = Z.shape
    _, C = w.shape
    _, F = v.shape
    # dL/dlogit = probs - onehot(y)
    grad_logit = probs.copy()
    grad_logit[np.arange(n), y] -= 1.0
    grad_logit /= n  # (n, C)

    # gradients for linear part
    dw = Z.T @ grad_logit + L2 * w  # (k, C)
    db = grad_logit.sum(axis=0)     # (C)

    # gradients for factor part
    # interaction gradient: dL/dinter = sum_c dL/dlogit_c  (n,)
    grad_inter = grad_logit.sum(axis=1)  # (n,)
    # For each factor f: dL/dv_{i,f} = grad_inter_i * Z_i * (s_f_i - Z_i * v_{i,f})
    # s = Z @ v  (n, F)
    s = Z @ v
    # We need Z_i * (s_f_i - Z_i * v_{i,f}) for each i, f.
    # This can be written as: (grad_inter[:, None] * Z)[:, :, None] * (s[:, None, :] - Z[:, :, None] * v[None, :, :])
    # But we can do it efficiently with loops over F (F is small) or vectorized.
    # Vectorized:
    # term = grad_inter[:, None] * (s[:, :, None] - Z[:, :, None] * v[None, :, :])  # (n, F, k)? Wait shapes.
    # Let's do: for each f, grad_v[:, f] = Z.T @ (grad_inter * (s[:, f] - Z * v[:, f])) + L2 * v[:, f]
    dv = np.zeros_like(v)
    for f in range(F):
        # s_f = s[:, f]  (n,)
        # v_f = v[:, f]  (k,)
        # Z * v_f -> (n, k)
        diff = s[:, f][:, None] - Z * v[None, :, f]  # (n, k)
        grad_v_f = (grad_inter[:, None] * diff).sum(axis=0) + L2 * v[:, f]
        dv[:, f] = grad_v_f
    return dw, db, dv

def train_fm(Z, y, C, F, rng, epochs, batch_size):
    n, k = Z.shape
    # init
    w = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    v = (rng.uniform(-1, 1, (k, F)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [w, b, v]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            z, yy = Z[idx], y[idx]
            logits = fm_forward(z, w, v, b)
            # softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dw, db, dv = fm_backward(z, yy, w, v, b, probs)
            grads = [dw, db, dv]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w, b, v

def evaluate(Z, y, w, b, v):
    logits = fm_forward(Z, w, v, b)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # build encoding on full training data (as per label-linear)
    enc = build_encoding(card, X)
    Z_full = encode_batch(X, enc)
    k = Z_full.shape[1]

    # split: last 20% for tuning
    n = len(y)
    split = int(0.8 * n)
    Z_train, Z_tune = Z_full[:split], Z_full[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_F, best_acc = None, -1.0
    # quick training for each F (fewer epochs for tuning)
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_train)))
    for F in CANDIDATE_F:
        w, b, v = train_fm(Z_train, y_train, C, F, rng, tune_epochs, BATCH)
        acc = evaluate(Z_tune, y_tune, w, b, v)
        if acc > best_acc:
            best_acc, best_F = acc, F

    # retrain on full data with best_F
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w, b, v = train_fm(Z_full, y, C, best_F, rng, final_epochs, BATCH)

    # compress to per-byte lookup tables
    linear_T = []  # list of (256, C)
    factor_T = []  # list of (256, F)
    off = 0
    for E in enc:
        kj = E.shape[1]
        linear_T.append((E @ w[off:off + kj]).astype(np.float32))
        factor_T.append((E @ v[off:off + kj]).astype(np.float32))
        off += kj
    linear_T = np.stack(linear_T)   # (d, 256, C)
    factor_T = np.stack(factor_T)   # (d, 256, F)

    # write weights.bin
    # header: magic, d, F, C (4 int32)
    header = struct.pack("<4i", 0x464D4C31, d, best_F, C)
    out = header
    out += linear_T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += factor_T.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights format: magic 0x464D4C31, int32 d, F, C;
 * float32 linear_T[d][256][C], bias_lin[C], factor_T[d][256][F]. */
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
    int d = hd[1], F = hd[2], C = hd[3];
    if (C < 1 || C > 256 || F < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * F);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear_T = (const float *)(hd + 4);
    const float *bias_lin = linear_T + (long)d * 256 * C;
    const float *factor_T = bias_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * F);
    float *sum_v2 = malloc(sizeof(float) * F);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias_lin[c];
        for (int f = 0; f < F; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *lin_row = linear_T + ((long)j * 256 + val) * C;
            for (int c = 0; c < C; c++) acc[c] += lin_row[c];

            const float *fac_row = factor_T + ((long)j * 256 + val) * F;
            for (int f = 0; f < F; f++) {
                float v = fac_row[f];
                sum_v[f] += v;
                sum_v2[f] += v * v;
            }
        }

        for (int f = 0; f < F; f++) {
            float inter = 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
            for (int c = 0; c < C; c++) acc[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```