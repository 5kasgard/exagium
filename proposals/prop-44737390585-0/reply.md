```hypothesis
Family: label-fm (factorization machine). Adult income has known feature interactions (e.g., education × occupation). A factorization machine captures all pairwise interactions with O(d·k) parameters (d=14, k=4 → ~18k floats) and adds only ~d·k operations per inference item. It should exceed label-linear accuracy (0.8524) while keeping CPU cost near linear. If dev accuracy ≤ 0.8524, interactions are not learnable with this capacity or the FM formulation is insufficient.
```

```python
"""Factorization machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the d=14 byte features as a linear weight + k-dim embedding (255=unseen).
Trains with Adam (lr=3e-3, L2=1e-4, batch=64) for min(100, ceil(600000/n)) epochs.
Tunes latent dimension k on last 20% of train.in (file order), then retrains on all data.
Exports: magic 0x464D4C31, d, k, C; float32 linear[d][256], embed[d][256][k], bias."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, k, seed, epochs):
    """Train FM on (X,y) for given epochs, return (W_lin, V, bias)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # Parameters: linear (d,256), embed (d,256,k), bias (1)
    W_lin = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.float32(0.0)
    params = [W_lin, V, bias[None]]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # Forward
            lin = np.sum(W_lin[np.arange(d)[:,None], xb.T], axis=0)  # (bsz,)
            sum_emb = np.zeros((bsz, k), np.float32)
            sum_sq = np.zeros((bsz, k), np.float32)
            for j in range(d):
                v = V[j, xb[:, j]]  # (bsz, k)
                sum_emb += v
                sum_sq += v * v
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq, axis=1)
            score = bias + lin + inter
            p = 1.0 / (1.0 + np.exp(-score))
            # Gradient: dL/dscore = p - y
            g = (p - yb).astype(np.float32)  # (bsz,)
            # Gradients
            g_bias = g.mean()
            # Linear grads: for each feature j, add g to W_lin[j, xb[:,j]]
            g_W = np.zeros_like(W_lin)
            for j in range(d):
                np.add.at(g_W[j], xb[:, j], g)
            g_W /= bsz
            g_W += L2 * W_lin
            # Embedding grads: for each j, g_V[j, xb[:,j], f] = g * (sum_emb[:,f] - V[j, xb[:,j], f])
            g_V = np.zeros_like(V)
            for j in range(d):
                v_j = V[j, xb[:, j]]  # (bsz, k)
                diff = sum_emb - v_j  # (bsz, k)
                for f in range(k):
                    np.add.at(g_V[j, :, f], xb[:, j], g * diff[:, f])
            g_V /= bsz
            g_V += L2 * V
            grads = [g_W, g_V, np.array([g_bias], np.float32)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params[0], params[1], params[2][0]

def evaluate_fm(X, y, W_lin, V, bias, k):
    n, d = X.shape
    lin = np.sum(W_lin[np.arange(d)[:,None], X.T], axis=0)
    sum_emb = np.zeros((n, k), np.float32)
    sum_sq = np.zeros((n, k), np.float32)
    for j in range(d):
        v = V[j, X[:, j]]
        sum_emb += v
        sum_sq += v * v
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq, axis=1)
    score = bias + lin + inter
    pred = (score > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    best_k, best_acc = 4, -1.0
    for k in K_CANDIDATES:
        W_lin, V, bias = train_fm(X_train, y_train, k, seed=SEED + k, epochs=epochs_tune)
        acc = evaluate_fm(X_val, y_val, W_lin, V, bias, k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # Retrain on all data with best_k
    W_lin, V, bias = train_fm(X, y, best_k, seed=SEED, epochs=epochs_full)
    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, best_k, C)
    linear_bytes = W_lin.astype("<f4").tobytes()
    embed_bytes = V.astype("<f4").tobytes()
    bias_bytes = np.array([bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(header + linear_bytes + embed_bytes + bias_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorization machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: one byte (class 0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 linear[d][256], embed[d][256][k], bias. */
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
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)d * 256 * k + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *embed = linear + (long)d * 256;
    const float *bias = embed + (long)d * 256 * k;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = *bias;
        // linear part
        for (int j = 0; j < d; j++) {
            score += linear[(long)j * 256 + xi[j]];
        }
        // interaction part
        for (int f = 0; f < k; f++) {
            float sum_emb = 0.0f, sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = embed[((long)j * 256 + xi[j]) * k + f];
                sum_emb += v;
                sum_sq += v * v;
            }
            score += 0.5f * (sum_emb * sum_emb - sum_sq);
        }
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```