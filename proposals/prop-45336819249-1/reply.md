```hypothesis
We propose a Factorization Machine family "label-fm" that augments a linear model with pairwise feature interactions via low-rank embeddings (rank k=8). Optdigits pixels are ordinal (0..16) and spatially correlated; degree-2 interactions capture stroke patterns that a linear model misses. The FM computes interactions in O(k*d) time (k=8, d=64), keeping inference cost close to label-linear. If dev accuracy does not exceed label-linear (0.9613) or instructions per item more than double (~10k), the hypothesis fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Model: scores = b + sum_j L[j,x_j] + 0.5*((sum_j E[j,x_j])^2 - sum_j E[j,x_j]^2) @ V
where L: linear weights (d,256,C), E: embeddings (d,256,k), V: interaction projection (k,C).
Hyperparameters chosen on last 20% of train (tuning split), then retrained on all train."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_EPOCHS = 10  # epochs for each grid configuration


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, k, rng):
    # He-ish initialization for L and E, small for V
    L = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    E = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)
    V = (rng.normal(0, 0.01, (k, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    return L, E, V, b


def forward_batch(X_batch, L, E, V, b):
    """X_batch: (B, d) uint8. Returns scores (B, C), embed_sum (B, k), embed_sq_sum (B, k)"""
    B = X_batch.shape[0]
    d = X_batch.shape[1]
    scores = b[None, :].repeat(B, axis=0)  # (B, C)
    embed_sum = np.zeros((B, E.shape[2]), np.float32)
    embed_sq_sum = np.zeros((B, E.shape[2]), np.float32)
    for j in range(d):
        v = X_batch[:, j]
        scores += L[j, v]
        e = E[j, v]
        embed_sum += e
        embed_sq_sum += e * e
    interaction = 0.5 * (embed_sum * embed_sum - embed_sq_sum)
    scores += interaction @ V
    return scores, embed_sum, embed_sq_sum


def loss_and_grad(X_batch, y_batch, L, E, V, b, l2):
    B = X_batch.shape[0]
    d = X_batch.shape[1]
    C = L.shape[2]
    k = E.shape[2]

    scores, embed_sum, embed_sq_sum = forward_batch(X_batch, L, E, V, b)
    # Softmax
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    # Cross-entropy gradient
    dS = probs
    dS[np.arange(B), y_batch] -= 1.0
    dS /= B

    # Gradients
    db = dS.sum(axis=0)
    dV = embed_sum.T @ dS * 0.5? Wait, interaction = 0.5*(embed_sum^2 - embed_sq_sum)
    # Let's compute d_interaction = dS @ V.T  (B, k)
    d_interaction = dS @ V.T  # (B, k)
    # derivative of interaction w.r.t embed_sum_f = embed_sum_f
    # derivative w.r.t embed_sq_sum_f = -0.5
    # So for each example i, factor f:
    # d_embed_sum_f = d_interaction[i,f] * embed_sum[i,f]
    # d_embed_sq_sum_f = d_interaction[i,f] * (-0.5)
    # But embed_sq_sum = sum_j e_j^2, so gradient for e_j is 2 * e_j * d_embed_sq_sum_f
    # Total gradient for e_j_f = d_embed_sum_f + 2 * e_j_f * d_embed_sq_sum_f
    # = d_interaction_f * (embed_sum_f - e_j_f)
    # We'll accumulate per (j, v)
    dL = np.zeros_like(L)
    dE = np.zeros_like(E)

    for j in range(d):
        v = X_batch[:, j]
        # dL: add dS for each example to L[j, v]
        np.add.at(dL[j], v, dS)
        # dE:
        e = E[j, v]  # (B, k)
        factor = d_interaction * (embed_sum - e)  # (B, k)
        np.add.at(dE[j], v, factor)

    # L2 regularization
    dL += l2 * L
    dE += l2 * E
    dV += l2 * V
    # no L2 on b

    # Loss for monitoring
    loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()
    return loss, dL, dE, dV, db


def adam_update(params, grads, states, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_eval(X_tr, y_tr, X_val, y_val, k, lr, l2, epochs, batch, rng):
    d = X_tr.shape[1]
    C = len(np.unique(y_tr))
    L, E, V, b = init_params(d, C, k, rng)
    params = [L, E, V, b]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X_tr.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, dL, dE, dV, db = loss_and_grad(Xb, yb, L, E, V, b, l2)
            grads = [dL, dE, dV, db]
            t += 1
            adam_update(params, grads, states, lr, t)
    # Evaluate on validation
    scores, _, _ = forward_batch(X_val, L, E, V, b)
    preds = scores.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, (L, E, V, b)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20%
    n_val = int(0.2 * n)
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None

    # Grid search
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                acc, params = train_eval(X_tr, y_tr, X_val, y_val, k, lr, l2, TUNE_EPOCHS, BATCH, rng)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (k, lr, l2)
                    best_params = params
    print(f"Best tuning: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    L, E, V, b = init_params(d, C, k, rng)
    params = [L, E, V, b]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            loss, dL, dE, dV, db = loss_and_grad(Xb, yb, L, E, V, b, l2)
            grads = [dL, dE, dV, db]
            t += 1
            adam_update(params, grads, states, lr, t)

    # Write weights.bin: magic, d, k, C, then b, L, E, V as float32 little-endian
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, k, C)
    out = header
    out += b.astype("<f4").tobytes()
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index (argmax).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 b[C], L[d*256*C], E[d*256*k], V[k*C]. */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 256 || d < 1 || d > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16
        + 4L * C
        + 4L * d * 256 * C
        + 4L * d * 256 * k
        + 4L * k * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *b = (const float *)(hd + 4);
    const float *L = b + C;
    const float *E = L + (long)d * 256 * C;
    const float *V = E + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *embed_sum = malloc(sizeof(float) * k);
    float *embed_sq_sum = malloc(sizeof(float) * k);
    float *interaction = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        memset(embed_sum, 0, k * sizeof(float));
        memset(embed_sq_sum, 0, k * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Ljv = L + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += Ljv[c];
            const float *Ejv = E + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float e = Ejv[f];
                embed_sum[f] += e;
                embed_sq_sum[f] += e * e;
            }
        }

        for (int f = 0; f < k; f++) {
            interaction[f] = 0.5f * (embed_sum[f] * embed_sum[f] - embed_sq_sum[f]);
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int f = 0; f < k; f++) sum += interaction[f] * V[(long)f * C + c];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(embed_sum); free(embed_sq_sum); free(interaction);
    if (x) free(x);
    return 0;
}
```