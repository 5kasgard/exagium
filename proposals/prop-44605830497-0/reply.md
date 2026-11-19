```hypothesis
We propose a factorization machine family "label-fm". FMs model pairwise feature interactions via low-rank latent vectors, which are highly effective for tabular data with mixed categorical/numeric features like Adult. The 14 features (each 256 bins) allow a small latent dimension (k=8) to capture interactions (e.g., education × occupation) that linear models miss. Training uses Adam on logistic loss with L2 regularization; hyperparameters (k, lr, L2) are chosen on the last 20% tuning split. Inference computes w0 + Σ w_j + ½((Σ v_j)² - Σ v_j²) in O(d·k) time, staying fast. If dev accuracy does not exceed label-linear (0.8524), interactions are not beneficial or training is inadequate.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the d=14 bytes as a categorical 0..255. Learns:
  w0 (bias), w[d,256] (linear weights), v[d,256,k] (latent vectors).
Binary classification via logistic loss. Hyperparameters (k, lr, L2) tuned on last 20% of train.in.
Exports binary format: magic 0x464D4C31, int32 d,k,C; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Fixed training configuration
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_score(w0, w, v, X, k):
    """Compute FM scores for batch X (n,d) uint8. Returns (n,) float32."""
    n, d = X.shape
    # Linear part
    lin = w0 + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    # Interaction part: 0.5 * ( (sum_v)^2 - sum_v_sq )
    # v shape (d,256,k) -> for each example, gather v[j, X[:,j], :] -> (n,d,k)
    v_gathered = v[np.arange(d), X.T]  # (d,n,k) after transpose? Let's do carefully
    # X is (n,d). We want for each example i, for each feature j, v[j, X[i,j], :]
    # Using advanced indexing:
    # v_gathered = v[np.arange(d)[:,None], X.T]  # (d, n, k)
    v_gathered = v[np.arange(d)[:, None], X.T]  # (d, n, k)
    sum_v = v_gathered.sum(axis=0)          # (n, k)
    sum_v_sq = (v_gathered ** 2).sum(axis=0)  # (n, k)
    inter = 0.5 * np.sum(sum_v ** 2 - sum_v_sq, axis=1)  # (n,)
    return (lin + inter).astype(np.float32)


def train_fm(X, y, k, lr, l2, epochs, rng):
    n, d = X.shape
    # Initialize parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]

            # Forward
            # Linear
            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (b,)
            # Interaction
            v_gathered = v[np.arange(d)[:, None], xb.T]      # (d, b, k)
            sum_v = v_gathered.sum(axis=0)                   # (b, k)
            sum_v_sq = (v_gathered ** 2).sum(axis=0)         # (b, k)
            inter = 0.5 * np.sum(sum_v ** 2 - sum_v_sq, axis=1)  # (b,)
            score = lin + inter
            prob = 1.0 / (1.0 + np.exp(-score))
            grad_score = (prob - yb).astype(np.float32)      # (b,)

            # Gradients
            # w0
            gw0 = grad_score.mean()
            # w: (d,256)
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], xb[:, j], grad_score)
            gw /= b
            # v: (d,256,k)
            gv = np.zeros_like(v)
            # For each feature j, we need (sum_v - v_j) * grad_score
            # sum_v is (b,k), v_gathered is (d,b,k)
            # diff = sum_v[:, None, :] - v_gathered   # (b, d, k) but we want (d,b,k)
            # Let's compute per feature
            for j in range(d):
                diff = sum_v - v_gathered[j]          # (b, k)
                # Multiply by grad_score[:, None]
                gv_j = diff * grad_score[:, None]     # (b, k)
                # Accumulate into gv[j, xb[:,j], :]
                np.add.at(gv[j], xb[:, j], gv_j)
            gv /= b

            # L2 regularization
            gw += l2 * w
            gv += l2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * (gv * gv)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)

    return w0, w, v


def evaluate(w0, w, v, X, y, k):
    scores = fm_score(w0, w, v, X, k)
    preds = (scores > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Determine epochs from formula
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                rng = np.random.default_rng(SEED)
                w0, w, v = train_fm(X_train, y_train, k, lr, l2, epochs, rng)
                acc = evaluate(w0, w, v, X_val, y_val, k)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (k, lr, l2)
                    best_params = (w0, w, v)

    print(f"Best config: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    w0, w, v = train_fm(X, y, k, lr, l2, epochs_full, rng)

    # Export weights
    # Format: magic, d, k, C, w0, w (d*256), v (d*256*k) all float32 little-endian
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, k, C)
    out += np.array(w0, dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index 0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 w0, w[d*256], v[d*256*k]. */
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
    if (C != 2 || k <= 0 || d <= 0 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float linear = w0;
        for (int j = 0; j < k; j++) sum_v[j] = 0.0f;
        float sum_v_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * k;
            for (int u = 0; u < k; u++) {
                float vju = vj[u];
                sum_v[u] += vju;
                sum_v_sq += vju * vju;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) inter += sum_v[u] * sum_v[u];
        inter = 0.5f * (inter - sum_v_sq);
        float score = linear + inter;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    return 0;
}
```