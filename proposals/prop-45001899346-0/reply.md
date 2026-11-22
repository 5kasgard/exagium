```hypothesis
Factorization machines (FM) capture pairwise feature interactions via low-rank embeddings, ideal for tabular data with mixed ordinal/categorical features already discretized to 256 bins. The model adds a single scalar interaction term to a linear logit, keeping inference at O(d·E) — faster than MLP/GBDT — while often matching their accuracy. If dev accuracy ≤ 0.8524 (label-linear) or instr/item ≥ label-mlp16 (2325), the hypothesis fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D4C31 ('FML1'), d, E, C; float32 bias, W1[d*256], Emb[d*256*E].
"""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (tuned on last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E_CANDIDATES = [2, 4, 8, 16, 32]  # embedding dimensions to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, d, E, C, rng, n_iter):
    """Train FM with Adam, return (bias, W1, Emb)."""
    # Parameters
    bias = np.zeros(1, np.float32)
    W1 = np.zeros((d, 256), np.float32)          # first-order weights (per class 1)
    Emb = rng.uniform(-0.01, 0.01, (d, 256, E)).astype(np.float32)

    # Adam state
    params = [bias, W1, Emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    for _ in range(n_iter):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            linear = bias[0] + np.sum(W1[np.arange(d), xb.T], axis=0)  # (bs,)
            sum_emb = np.sum(Emb[np.arange(d), xb.T], axis=0)          # (bs, E)
            sum_emb_sq = np.sum(Emb[np.arange(d), xb.T] ** 2, axis=0)  # (bs, E)
            interaction = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)  # (bs,)
            logit = linear + interaction                               # (bs,)

            # Binary cross-entropy gradient
            p = 1.0 / (1.0 + np.exp(-logit))          # (bs,)
            grad_logit = (p - yb) / bs                # (bs,)

            # Gradients
            g_bias = np.sum(grad_logit)
            g_W1 = np.zeros_like(W1)
            g_Emb = np.zeros_like(Emb)

            # First-order gradient
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_W1[j], vals, grad_logit)

            # Embedding gradients
            # d(interaction)/d(emb_{j,v,e}) = grad_logit * (sum_emb_e - emb_{j,v,e})
            for j in range(d):
                vals = xb[:, j]
                for e in range(E):
                    sum_e = sum_emb[:, e]
                    emb_je = Emb[j, vals, e]
                    g = grad_logit * (sum_e - emb_je)
                    np.add.at(g_Emb[j, :, e], vals, g)

            # L2 regularization
            g_W1 += L2 * W1
            g_Emb += L2 * Emb

            grads = [np.array([g_bias], np.float32), g_W1, g_Emb]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return bias, W1, Emb


def eval_fm(X, y, bias, W1, Emb):
    d, E = W1.shape[0], Emb.shape[2]
    linear = bias[0] + np.sum(W1[np.arange(d), X.T], axis=0)
    sum_emb = np.sum(Emb[np.arange(d), X.T], axis=0)
    sum_emb_sq = np.sum(Emb[np.arange(d), X.T] ** 2, axis=0)
    interaction = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)
    logit = linear + interaction
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # Tuning split: last 20% in file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_E, best_acc = 0, -1.0
    n_iter = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))

    for E in E_CANDIDATES:
        bias, W1, Emb = train_fm(X_tr, y_tr, d, E, C, rng, n_iter)
        acc = eval_fm(X_val, y_val, bias, W1, Emb)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on full train with best_E
    n_iter_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, W1, Emb = train_fm(X, y, d, best_E, C, rng, n_iter_full)

    # Write weights.bin
    # Header: magic, d, E, C
    out = struct.pack("<4i", 0x464D4C31, d, best_E, C)
    out += bias.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += Emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (feature indices). Output record: 1 byte (class 0/1).
 * Weights format (little-endian): int32 magic=0x464D4C31, d, E, C;
 * float32 bias, W1[d*256], Emb[d*256*E].
 */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C != 2 || E < 1 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * E);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W1 = bias + 1;
    const float *Emb = W1 + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(E, sizeof(float));
    float *sum_emb_sq = calloc(E, sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias[0];
        memset(sum_emb, 0, E * sizeof(float));
        memset(sum_emb_sq, 0, E * sizeof(float));

        for (int j = 0; j < d; j++) {
            int idx = xi[j];
            linear += W1[(long)j * 256 + idx];
            const float *emb_ptr = Emb + ((long)j * 256 + idx) * E;
            for (int e = 0; e < E; e++) {
                float v = emb_ptr[e];
                sum_emb[e] += v;
                sum_emb_sq[e] += v * v;
            }
        }

        float interaction = 0.0f;
        for (int e = 0; e < E; e++) {
            interaction += sum_emb[e] * sum_emb[e] - sum_emb_sq[e];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        float prob = 1.0f / (1.0f + expf(-logit));
        out_byte = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }

    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```