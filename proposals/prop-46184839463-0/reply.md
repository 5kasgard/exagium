```hypothesis
Family: label-fm (Factorization Machine with rank-4 embeddings, shared across classes).
Why: Adult has strong feature interactions (marital×relationship, education×occupation, capital-gain×hours) that linear models miss. FM captures all pairwise interactions in O(d·k) via the kernel trick (d=14, k=4 → ~56 mul-adds/item), keeping inference ultra-fast while adding expressive power.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000, the FM overhead isn't justified.
```
```python
"""Factorization Machine (rank k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, k, C; float32 w[C][d][256], V[d][256][k], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on tuning split (last 20% of train.in)
K_CHOICES = [2, 4, 8]
L2_CHOICES = [1e-4, 1e-3, 1e-2]
LR_CHOICES = [1e-3, 3e-3]

SEED = 1
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


def split_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    t = int(n * (1 - tune_frac))
    return X[:t], y[:t], X[t:], y[t:]


def train_fm(X, y, C, k, lr, l2, rng, epochs, batch):
    n, d = X.shape
    # params: w[C, d, 256], V[d, 256, k], bias[C]
    w = (rng.uniform(-0.01, 0.01, (C, d, 256)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [w, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute one-hot indices for speed
    X_idx = X.astype(np.int32)  # [n, d]

    def forward(idx_batch):
        # idx_batch: [b, d]
        b = idx_batch.shape[0]
        # linear: [b, C]
        lin = bias.copy()
        for c in range(C):
            for j in range(d):
                lin[:, c] += w[c, j, idx_batch[:, j]]
        # FM: sum_V[f] = sum_j V[j, x_j, f]
        sum_V = np.zeros((b, k), np.float32)
        sum_V_sq = np.zeros((b, k), np.float32)
        for j in range(d):
            v = V[j, idx_batch[:, j], :]  # [b, k]
            sum_V += v
            sum_V_sq += v * v
        inter = 0.5 * (sum_V * sum_V - sum_V_sq).sum(axis=1)  # [b]
        logits = lin + inter[:, None]  # [b, C]
        return logits

    def loss_grad(idx_batch, y_batch):
        b = idx_batch.shape[0]
        logits = forward(idx_batch)
        # softmax + cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(b), y_batch] -= 1.0
        probs /= b  # [b, C]

        # grads
        gw = np.zeros_like(w)
        gV = np.zeros_like(V)
        gbias = probs.sum(axis=0)

        # linear grads
        for c in range(C):
            for j in range(d):
                vals = idx_batch[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        gw[c, j, v] = probs[mask, c].sum()

        # FM grads
        # dL/d(inter) = probs.sum(axis=1)  [b]
        dinter = probs.sum(axis=1)  # [b]
        # inter = 0.5 * (sum_V^2 - sum_V_sq).sum(f)
        # dinter/dV[j,v,f] = (sum_V[f] - V[j,v,f]) * dinter
        sum_V = np.zeros((b, k), np.float32)
        for j in range(d):
            sum_V += V[j, idx_batch[:, j], :]
        for j in range(d):
            v_batch = V[j, idx_batch[:, j], :]  # [b, k]
            grad_v = (sum_V - v_batch) * dinter[:, None]  # [b, k]
            for v in range(256):
                mask = (idx_batch[:, j] == v)
                if mask.any():
                    gV[j, v, :] = grad_v[mask].sum(axis=0)

        # L2 regularization
        gw += l2 * w
        gV += l2 * V
        return [gw, gV, gbias]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X_idx[idx], y[idx]
            grads = loss_grad(xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w, V, bias


def eval_acc(X, y, w, V, bias, C):
    n, d = X.shape
    correct = 0
    for i in range(0, n, 1024):
        xb = X[i:i+1024]
        yb = y[i:i+1024]
        b = xb.shape[0]
        lin = bias.copy()
        for c in range(C):
            for j in range(d):
                lin[:, c] += w[c, j, xb[:, j]]
        sum_V = np.zeros((b, V.shape[2]), np.float32)
        sum_V_sq = np.zeros((b, V.shape[2]), np.float32)
        for j in range(d):
            v = V[j, xb[:, j], :]
            sum_V += v
            sum_V_sq += v * v
        inter = 0.5 * (sum_V * sum_V - sum_V_sq).sum(axis=1)
        logits = lin + inter[:, None]
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Tuning split
    X_tr, y_tr, X_va, y_va = split_tune(X, y)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X_tr.shape[0]))

    best_acc = -1
    best_cfg = None
    best_params = None

    for k in K_CHOICES:
        for l2 in L2_CHOICES:
            for lr in LR_CHOICES:
                w, V, bias = train_fm(X_tr, y_tr, C, k, lr, l2, rng, epochs, BATCH)
                acc = eval_acc(X_va, y_va, w, V, bias, C)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (k, lr, l2)
                    best_params = (w.copy(), V.copy(), bias.copy())
                print(f"  k={k} l2={l2} lr={lr} -> {acc:.4f}", file=sys.stderr)

    k, lr, l2 = best_cfg
    print(f"Best: k={k} lr={lr} l2={l2} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    w, V, bias = train_fm(X, y, C, k, lr, l2, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0])), BATCH)

    # Pack weights: magic, d, k, C; w[C][d][256], V[d][256][k], bias[C] (all float32 LE)
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, k, C)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (rank k) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax).
 * Weights layout: int32 magic=0x464D4C31, d, k, C;
 *   float32 w[C][d][256], V[d][256][k], bias[C] (all little-endian). */
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
    if (C < 1 || C > 256 || d < 1 || k < 1 ||
        nw != 16 + 4L * ((long)C * d * 256 + (long)d * 256 * k + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *w = (const float *)(hd + 4);
    const float *V = w + (long)C * d * 256;
    const float *bias = V + (long)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * k);
    float *sum_V_sq = malloc(sizeof(float) * k);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                logits[c] += w[((long)c * d + j) * 256 + v];
            }
        }

        // FM part: sum_V[f] = sum_j V[j, xi[j], f]
        for (int f = 0; f < k; f++) { sum_V[f] = 0.0f; sum_V_sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Vjv = V + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = Vjv[f];
                sum_V[f] += val;
                sum_V_sq[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            float inter = 0.5f * (sum_V[f] * sum_V[f] - sum_V_sq[f]);
            for (int c = 0; c < C; c++) logits[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```