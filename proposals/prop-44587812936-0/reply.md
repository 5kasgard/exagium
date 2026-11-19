```hypothesis
Family: label-fm (Factorization Machine with per-class embeddings).
Why: Adult data has strong feature interactions (e.g., education×occupation); FM captures pairwise interactions efficiently via low-rank embeddings, outperforming linear models with minimal overhead. The 14-field, 256-value structure suits FM's embedding tables.
Failure sign: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the family does not justify its complexity.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Trains a per-class FM with embedding dimension k chosen on tuning split (last 20% of train).
Exports: magic 0x464D3132, d, k, C; float32 bias[C], linear[C][d][256], embed[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_GRID = [0.01, 0.003]
K_GRID = [4, 8, 16]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(scores, y):
    return np.mean(np.argmax(scores, axis=0) == y)


def forward(X, bias, linear, embed, C, d, k):
    """X: (B, d) uint8 -> scores: (C, B) float32"""
    B = X.shape[0]
    # linear part: bias[C] + sum_j linear[c, j, X[:,j]]
    lin_scores = bias[:, None] + np.sum(linear[:, np.arange(d), X.T], axis=1)  # (C, B)
    scores = lin_scores.copy()
    # FM interaction per class
    for c in range(C):
        emb = embed[c, np.arange(d), X.T]  # (d, B, k)
        sum_v = np.sum(emb, axis=0)  # (B, k)
        sum_v_sq = np.sum(emb * emb, axis=0)
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)  # (B,)
        scores[c] += interaction
    return scores


def backward(X, y, scores, bias, linear, embed, C, d, k, L2):
    """Compute gradients for a batch. Returns (gbias, glinear, gembed)."""
    B = X.shape[0]
    # softmax gradient
    max_s = np.max(scores, axis=0, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=0, keepdims=True)
    probs[y, np.arange(B)] -= 1.0
    probs /= B  # (C, B)

    gbias = np.sum(probs, axis=1)  # (C,)
    glinear = np.zeros_like(linear)
    gembed = np.zeros_like(embed)

    for c in range(C):
        gc = probs[c]  # (B,)
        # linear gradients
        for j in range(d):
            glinear[c, j] += np.bincount(X[:, j], weights=gc, minlength=256)
        # embedding gradients
        emb = embed[c, np.arange(d), X.T]  # (d, B, k)
        sum_v = np.sum(emb, axis=0)  # (B, k)
        for j in range(d):
            v = X[:, j]
            grad_emb = gc[:, None] * (sum_v - emb[j])  # (B, k)
            for u in range(k):
                np.add.at(gembed[c, j, :, u], v, grad_emb[:, u])

    # L2 regularization
    glinear += L2 * linear
    gembed += L2 * embed
    return gbias, glinear, gembed


def train_model(X, y, X_val, y_val, C, d, k, lr, epochs, rng):
    """Train FM on (X,y), return (bias, linear, embed, val_acc)."""
    bias = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    embed = (rng.normal(0, 0.01, (C, d, 256, k))).astype(np.float32)

    # Adam state
    params = [bias, linear, embed]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            scores = forward(xb, bias, linear, embed, C, d, k)
            gbias, glinear, gembed = backward(xb, yb, scores, bias, linear, embed, C, d, k, L2)
            grads = [gbias, glinear, gembed]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_scores = forward(X_val, bias, linear, embed, C, d, k)
    val_acc = accuracy(val_scores, y_val)
    return bias, linear, embed, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for k in K_GRID:
        for lr in LR_GRID:
            # deterministic seed per config
            cfg_rng = np.random.default_rng(SEED + k * 100 + int(lr * 10000))
            bias, linear, embed, val_acc = train_model(X_tr, y_tr, X_val, y_val, C, d, k, lr, epochs, cfg_rng)
            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg = (k, lr)
                best_params = (bias.copy(), linear.copy(), embed.copy())
            print(f"k={k} lr={lr} val_acc={val_acc:.4f}", file=sys.stderr)

    k, lr = best_cfg
    print(f"Best: k={k} lr={lr} val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    rng_full = np.random.default_rng(SEED + k * 100 + int(lr * 10000) + 12345)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, linear, embed, _ = train_model(X, y, X, y, C, d, k, lr, epochs_full, rng_full)

    # write weights
    MAGIC = 0x464D3132  # "FM12"
    out = struct.pack("<4i", MAGIC, d, k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weights format: magic 0x464D3132, int32 d, k, C; float32 bias[C], linear[C][d][256], embed[C][d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16
        + (long)C * 4
        + (long)C * d * 256 * 4
        + (long)C * d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *sum_v = malloc(k * sizeof(float));
    float *sum_v_sq = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // linear part
        for (int c = 0; c < C; c++) {
            const float *lin_c = linear + (long)c * d * 256;
            for (int j = 0; j < d; j++) {
                scores[c] += lin_c[(long)j * 256 + xi[j]];
            }
        }

        // FM interaction per class
        for (int c = 0; c < C; c++) {
            memset(sum_v, 0, k * sizeof(float));
            memset(sum_v_sq, 0, k * sizeof(float));
            const float *emb_c = embed + (long)c * d * 256 * k;
            for (int j = 0; j < d; j++) {
                const float *v = emb_c + ((long)j * 256 + xi[j]) * k;
                for (int u = 0; u < k; u++) {
                    float val = v[u];
                    sum_v[u] += val;
                    sum_v_sq[u] += val * val;
                }
            }
            float inter = 0.0f;
            for (int u = 0; u < k; u++) {
                inter += sum_v[u] * sum_v[u] - sum_v_sq[u];
            }
            scores[c] += 0.5f * inter;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_v); free(sum_v_sq);
    return 0;
}
```