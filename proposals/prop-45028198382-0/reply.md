```hypothesis
Factorization Machines (FM) model pairwise feature interactions in O(d·k) time (d=14 features, k=latent dim), capturing critical Adult interactions like education×occupation and marital-status×relationship that linear models miss. With class-specific factors (C=2), inference stays ~14·k·2 ops. Tuning k on the last 20% of train.in should beat label-linear's 0.8524 dev accuracy while keeping Ir/item < 2000. If dev accuracy ≤ 0.8524 or Ir/item > 2000, the FM does not justify its complexity.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal as z-scored, categorical as one-hot over training values.
Exports: magic 0x464D4231 ("FMB1"), d, k, C; float32 linear_T[d][256][C], linear_b[C]; factor_T[d][256][k][C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [4, 8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of (256, nvals) encoders and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """X: (n, d) uint8 -> Z: (n, total_dim) float32, also return feature indices for FM."""
    n, d = X.shape
    Z_list = []
    feat_idx = []  # for each column in Z, which feature (0..d-1) and which value (0..255)
    for j, E in enumerate(enc):
        Z_list.append(E[X[:, j]])
        nvals = E.shape[1]
        for v in range(nvals):
            feat_idx.append((j, v))
    return np.concatenate(Z_list, axis=1).astype(np.float32), feat_idx


def fm_forward(Z, idx, W_lin, b_lin, V, k, C):
    """Z: (batch, dim), idx: list of (feat, val) per column. Returns logits (batch, C)."""
    batch = Z.shape[0]
    # Linear part
    lin = Z @ W_lin + b_lin  # (batch, C)
    # FM interaction part: class-specific factors V[dim, k, C]
    # For each class, compute 0.5 * sum_f ( (sum_i v_if)^2 - sum_i v_if^2 )
    # where v_if = V[col_i, f, c] and col_i is the active column for feature j.
    # Since Z is one-hot per feature-group, we can sum over active columns.
    # But Z is dense; we'll use the fact that each feature has exactly one active value.
    # Reconstruct per-feature sums from Z @ V_f for each factor f and class c.
    # V shape: (dim, k, C)
    # For each factor f and class c: sum_i Z_i * V_i,f,c  -> (batch,)
    # Then 0.5 * (sum^2 - sum(Z_i * V_i,f,c^2))
    # Since Z is 0/1, sum(Z_i * V^2) = sum over active of V^2.
    inter = np.zeros((batch, C), dtype=np.float32)
    for c in range(C):
        for f in range(k):
            Vf = V[:, f, c]  # (dim,)
            sum_v = Z @ Vf   # (batch,)
            sum_v2 = Z @ (Vf * Vf)
            inter[:, c] += 0.5 * (sum_v * sum_v - sum_v2)
    return lin + inter


def train_fm(Z, y, idx, k, C, rng, n_epochs):
    dim = Z.shape[1]
    # Initialize
    scale = math.sqrt(6.0 / dim)
    W_lin = (rng.uniform(-1, 1, (dim, C)) * scale).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (dim, k, C)) * math.sqrt(6.0 / dim)).astype(np.float32)
    params = [W_lin, b_lin, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = Z.shape[0]
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            z, yy = Z[batch_idx], y[batch_idx]
            # Forward
            logits = fm_forward(z, idx, W_lin, b_lin, V, k, C)
            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # (batch, C)
            # Gradients
            grad_W_lin = z.T @ g + L2 * W_lin
            grad_b_lin = g.sum(0)
            # FM gradients wrt V
            # d/dV_{i,f,c} inter_c = 0.5 * (2 * sum_v * z_i - 2 * z_i * V_{i,f,c})
            #                        = z_i * (sum_v - V_{i,f,c})
            # where sum_v = sum_j z_j * V_{j,f,c}
            grad_V = np.zeros_like(V)
            for c in range(C):
                for f in range(k):
                    Vf = V[:, f, c]
                    sum_v = z @ Vf  # (batch,)
                    # For each sample, for each active feature i: grad_V[i,f,c] += z_i * (sum_v - Vf[i])
                    # Since z is 0/1, this is: for each sample, for each active column i: (sum_v[s] - Vf[i])
                    # Vectorized: grad_V[:,f,c] += z.T @ (sum_v[:,None] - Vf[None,:])  -- wrong, z is (batch, dim)
                    # Correct: grad_V[i,f,c] = sum_s z_{s,i} * (sum_v[s] - Vf[i])
                    # = (z.T @ sum_v) - (z.sum(0) * Vf)
                    grad_V[:, f, c] = z.T @ sum_v - (z.sum(0) * Vf)
            grad_V += L2 * V
            grads = [grad_W_lin, grad_b_lin, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W_lin, b_lin, V


def eval_acc(Z, y, idx, W_lin, b_lin, V, k, C):
    logits = fm_forward(Z, idx, W_lin, b_lin, V, k, C)
    pred = logits.argmax(1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    enc = build_encoders(X, types)
    Z, idx = encode(X, enc)

    # Split last 20% for tuning k
    n = len(y)
    split = int(n * 0.8)
    Z_tr, Z_val = Z[:split], Z[split:]
    y_tr, y_val = y[:split], y[split:]

    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_tr)))
        W_lin, b_lin, V = train_fm(Z_tr, y_tr, idx, k, C, np.random.default_rng(SEED), n_epochs)
        acc = eval_acc(Z_val, y_val, idx, W_lin, b_lin, V, k, C)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # Retrain on full data with best_k
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W_lin, b_lin, V = train_fm(Z, y, idx, best_k, C, np.random.default_rng(SEED), n_epochs)

    # Export weights
    # Magic, d, k, C
    out = struct.pack("<4i", 0x464D4231, d, best_k, C)
    # Linear tables: per feature (d), 256 values, C classes
    # We have W_lin of shape (total_dim, C). Need to reshape to (d, 256, C)
    # But encoders have different nvals per feature. We'll create dense 256 tables.
    off = 0
    lin_T = np.zeros((d, 256, C), dtype=np.float32)
    for j, E in enumerate(enc):
        nvals = E.shape[1]
        lin_T[j, :nvals] = W_lin[off:off + nvals]
        off += nvals
    out += lin_T.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    # Factor tables: (d, 256, k, C)
    fac_T = np.zeros((d, 256, best_k, C), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        nvals = E.shape[1]
        fac_T[j, :nvals] = V[off:off + nvals]
        off += nvals
    out += fac_T.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Format: magic 0x464D4231 ("FMB1"), int32 d, k, C; float32 lin_T[d][256][C], lin_b[C], fac_T[d][256][k][C]. */
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
    if (nw < 16 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L
        + 4L * d * 256 * C          // lin_T
        + 4L * C                    // lin_b
        + 4L * d * 256 * k * C;     // fac_T
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *lin_T = (const float *)(hd + 4);
    const float *lin_b = lin_T + (long)d * 256 * C;
    const float *fac_T = lin_b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * k * C);
    float *sum_v2 = malloc(sizeof(float) * k * C);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = lin_b[c];
        for (int j = 0; j < d; j++) {
            const float *t = lin_T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }

        // FM interaction part
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < k; f++) {
                sum_v[c * k + f] = 0.0f;
                sum_v2[c * k + f] = 0.0f;
            }
        }
        for (int j = 0; j < d; j++) {
            const float *v = fac_T + ((long)j * 256 + xi[j]) * k * C;
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < k; f++) {
                    float val = v[c * k + f];
                    sum_v[c * k + f] += val;
                    sum_v2[c * k + f] += val * val;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < k; f++) {
                float sv = sum_v[c * k + f];
                inter += 0.5f * (sv * sv - sum_v2[c * k + f]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```