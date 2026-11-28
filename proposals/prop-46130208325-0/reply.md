```hypothesis
Family: label-fm (Factorization Machine with per-feature-value embeddings).
Why: Adult data has mixed ordinal/categorical features; pairwise interactions (e.g., education × occupation) are known to boost accuracy. A low-rank FM (K=4) adds only ~14k parameters over label-linear, keeping inference fast (O(d·K) per item) while capturing 2nd-order effects. The tuning split selects K ∈ {2,4,8,16} automatically.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-mlp16 (~2300 Ir/item), the family does not justify its complexity.
```

```python
"""Factorization Machine family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameter K (interaction rank) chosen on last 20% of train.in, then retrained on all data.
Exported weights: magic 0x464D4C31, d, K, C; float32 L[d][256][C], E[d][256][K], b1[C], w_inter[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [2, 4, 8, 16]          # tuned on the last 20% of train.in
TUNE_EPOCHS = 5                    # quick tuning epochs


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_embeddings(card, X):
    """Return list of (256, dim) embedding matrices for each feature (ordinal: 1-d, categorical: one-hot)."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                              # (256, n_cat)
    return enc


def train_fm(X, y, C, K, epochs, rng, l2=L2, lr=LR, batch=BATCH):
    """Train FM on given data, return parameters (L, E, b1, w_inter)."""
    n, d = X.shape
    # Linear embeddings: L[j][v][c]  (d, 256, C)
    L = np.zeros((d, 256, C), dtype=np.float32)
    # Interaction embeddings: E[j][v][k] (d, 256, K)
    E = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    b1 = np.zeros(C, dtype=np.float32)
    w_inter = np.zeros(C, dtype=np.float32)

    params = [L, E, b1, w_inter]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]                     # (b, d)
            yb = y[idx]                     # (b,)
            b = len(idx)

            # Gather embeddings
            L_batch = L[np.arange(d), xb.T]         # (d, b, C)
            E_batch = E[np.arange(d), xb.T]         # (d, b, K)

            # Linear part: sum over features
            linear = L_batch.sum(axis=0)            # (b, C)

            # Interaction term I = 0.5*(|sum e|^2 - sum |e|^2)
            sum_e = E_batch.sum(axis=0)             # (b, K)
            sum_e_sq = np.sum(sum_e * sum_e, axis=1)        # (b,)
            sum_sq_e = np.sum(E_batch * E_batch, axis=(0,1))# (b,)  -- wait, need per sample
            # Actually sum over d and k: for each sample, sum_{j,k} E_batch[j, sample, k]^2
            sum_sq_e = np.sum(E_batch * E_batch, axis=(0, 2)) # (b,)
            I = 0.5 * (sum_e_sq - sum_sq_e)         # (b,)

            logits = linear + b1 + w_inter * I[:, None]   # (b, C)

            # Softmax cross-entropy gradient
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(b), yb] -= 1.0
            g = (probs / b).astype(np.float32)           # (b, C)

            # Gradients
            # dL: for each feature j, add g to L[j][xb[:,j]]
            grad_L = np.zeros_like(L)
            for j in range(d):
                np.add.at(grad_L[j], xb[:, j], g)        # (256, C)

            # dE: for each feature j, grad = g_sum * (sum_e - e_j)
            g_sum = (g * w_inter).sum(axis=1)            # (b,)
            grad_E = np.zeros_like(E)
            for j in range(d):
                e_j = E_batch[j]                         # (b, K)
                diff = sum_e - e_j                       # (b, K)
                grad_e = g_sum[:, None] * diff           # (b, K)
                np.add.at(grad_E[j], xb[:, j], grad_e)   # (256, K)

            grad_b1 = g.sum(axis=0)
            grad_w_inter = (g * I[:, None]).sum(axis=0)

            grads = [grad_L, grad_E, grad_b1, grad_w_inter]

            # Adam update with L2 on L, E, w_inter (not on b1)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if i != 2:  # no L2 on bias
                    gr = gr + l2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return L, E, b1, w_inter


def evaluate(X, y, C, L, E, b1, w_inter):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        linear = L[np.arange(X.shape[1]), xi].sum(axis=0)
        sum_e = E[np.arange(X.shape[1]), xi].sum(axis=0)
        sum_e_sq = np.sum(sum_e * sum_e)
        sum_sq_e = np.sum(E[np.arange(X.shape[1]), xi] ** 2)
        I = 0.5 * (sum_e_sq - sum_sq_e)
        logits = linear + b1 + w_inter * I
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_K = K_CHOICES[0]
    best_acc = -1.0
    for K in K_CHOICES:
        rng = np.random.default_rng(SEED)
        L, E, b1, w_inter = train_fm(X_train, y_train, C, K, TUNE_EPOCHS, rng)
        acc = evaluate(X_tune, y_tune, C, L, E, b1, w_inter)
        if acc > best_acc:
            best_acc = acc
            best_K = K

    # Retrain on all data with best_K
    rng = np.random.default_rng(SEED)
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    L, E, b1, w_inter = train_fm(X, y, C, best_K, full_epochs, rng)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, best_K, C)
    out = header
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += w_inter.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: 1 byte class index (argmax). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)
        + (size_t)d * 256 * K * sizeof(float)
        + (size_t)C * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *E = L + (size_t)d * 256 * C;
    const float *b1 = E + (size_t)d * 256 * K;
    const float *w_inter = b1 + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *linear = malloc(sizeof(float) * C);
    float *sum_e = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* linear part */
        for (int c = 0; c < C; c++) linear[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *lj = L + ((size_t)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) linear[c] += lj[c];
        }

        /* interaction term */
        for (int kk = 0; kk < K; kk++) sum_e[kk] = 0.0f;
        float sum_sq_e = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((size_t)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = ej[kk];
                sum_e[kk] += v;
                sum_sq_e += v * v;
            }
        }
        float norm_sum_sq = 0.0f;
        for (int kk = 0; kk < K; kk++) norm_sum_sq += sum_e[kk] * sum_e[kk];
        float I = 0.5f * (norm_sum_sq - sum_sq_e);

        /* final logits */
        for (int c = 0; c < C; c++) linear[c] += w_inter[c] * I;

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (linear[c] > linear[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```