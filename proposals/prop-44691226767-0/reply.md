```hypothesis
Factorization Machine (FM) with low-rank embeddings (k=2 or 4) captures pairwise feature interactions
missing in label-linear. Adult has many categorical×categorical interactions (education×occupation,
marital×relationship, etc.) that FM models compactly. Training uses Adam on logistic loss with L2
on the 80/20 tuning split to pick k, lr, L2; then retrains on all data. If dev accuracy ≤ 0.8524
or inference cost exceeds label-linear by >2×, the approach does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_score(X, bias, W, V, k):
    """X: (n, d) uint8, W: (d, 256), V: (d, 256, k) -> scores (n,)"""
    n, d = X.shape
    lin = np.sum(W[np.arange(d), X.T], axis=0)  # (n,)
    if k:
        # pairwise dot products: sum_{j<j'} <V[j, x_j], V[j', x_j']>
        # compute sum of embeddings per example: (n, k)
        sumV = np.sum(V[np.arange(d), X.T].transpose(1,0,2), axis=1)  # (n, k)
        # sum of squares per example: (n, k)
        sumV2 = np.sum(V[np.arange(d), X.T].transpose(1,0,2)**2, axis=1)  # (n, k)
        inter = 0.5 * np.sum(sumV**2 - sumV2, axis=1)  # (n,)
    else:
        inter = 0.0
    return bias + lin + inter

def train_fm(X, y, k, lr, l2, rng):
    n, d = X.shape
    # init
    bias = 0.0
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32) if k else None
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    if k:
        mV, vV = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            scores = fm_score(xb, bias, W, V, k)
            p = sigmoid(scores)
            # gradients
            diff = (p - yb).astype(np.float32)  # (B,)
            # bias
            g_bias = diff.mean()
            # linear
            gW = np.zeros_like(W)
            for j in range(d):
                vals = xb[:, j]
                for c in range(256):
                    mask = (vals == c)
                    if mask.any():
                        gW[j, c] = diff[mask].mean()
            gW += l2 * W
            # embeddings
            if k:
                gV = np.zeros_like(V)
                # sum of embeddings for other features per example
                # precompute sumV for batch: (B, k)
                sumV_batch = np.zeros((len(xb), k), np.float32)
                for j in range(d):
                    sumV_batch += V[j, xb[:, j]]
                for j in range(d):
                    vj = V[j, xb[:, j]]  # (B, k)
                    # grad for V[j, x_j] = diff * (sumV_batch - vj)
                    gj = diff[:, None] * (sumV_batch - vj)  # (B, k)
                    # average per category
                    for c in range(256):
                        mask = (xb[:, j] == c)
                        if mask.any():
                            gV[j, c] = gj[mask].mean(axis=0)
                gV += l2 * V
            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1**t)
            v_hat = v_bias / (1 - beta2**t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # W
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            W -= lr * (mW / (1 - beta1**t)) / (np.sqrt(vW / (1 - beta2**t)) + eps)
            if k:
                mV = beta1 * mV + (1 - beta1) * gV
                vV = beta2 * vV + (1 - beta2) * gV * gV
                V -= lr * (mV / (1 - beta1**t)) / (np.sqrt(vV / (1 - beta2**t)) + eps)
    return bias, W, V

def val_accuracy(X, y, bias, W, V, k):
    scores = fm_score(X, bias, W, V, k)
    pred = (scores > 0).astype(np.int64)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = len(X)
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # hyperparameter grid
    grid = [
        {"k": 2, "lr": 0.01, "l2": 0.0},
        {"k": 2, "lr": 0.005, "l2": 0.001},
        {"k": 4, "lr": 0.01, "l2": 0.0},
        {"k": 4, "lr": 0.005, "l2": 0.001},
    ]
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for cfg in grid:
        rng = np.random.default_rng(SEED + hash((cfg["k"], cfg["lr"], cfg["l2"])) % 2**32)
        bias, W, V = train_fm(X_tr, y_tr, cfg["k"], cfg["lr"], cfg["l2"], rng)
        acc = val_accuracy(X_val, y_val, bias, W, V, cfg["k"])
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (bias, W, V)
    # retrain on full data with best config
    rng = np.random.default_rng(SEED + 12345)
    bias, W, V = train_fm(X, y, best_cfg["k"], best_cfg["lr"], best_cfg["l2"], rng)

    # pack weights: magic, d, k, C, bias, W[d][256], V[d][256][k]
    k = best_cfg["k"]
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", bias)
    out += W.astype("<f4").tobytes()
    if k:
        out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t*)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || (k != 0 && k != 2 && k != 4)) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + 4LL*d*256 + (k ? 4LL*d*256*k : 0);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float*)(hd + 4);
    float bias = *p++;                     // 1
    const float *W = p; p += d * 256;      // d*256
    const float *V = k ? p : NULL;         // d*256*k

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = bias;
        // linear part
        for (int j = 0; j < d; j++) score += W[j*256 + xi[j]];
        // pairwise interactions
        if (k) {
            float sumV[4] = {0}, sumV2[4] = {0}; // k <= 4
            for (int j = 0; j < d; j++) {
                const float *v = V + (j*256 + xi[j]) * k;
                for (int l = 0; l < k; l++) {
                    float val = v[l];
                    sumV[l] += val;
                    sumV2[l] += val * val;
                }
            }
            for (int l = 0; l < k; l++) score += 0.5f * (sumV[l]*sumV[l] - sumV2[l]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```