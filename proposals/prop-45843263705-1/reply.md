```hypothesis
We propose a polynomial logistic regression family (label-poly2). It expands each 8×8 digit into degree‑2 polynomial features (bias, linear, and all quadratic terms) and trains a multinomial logistic regression with L2 regularization. This captures pixel interactions that are crucial for digit recognition, while remaining a convex model that trains quickly and deterministically. Inference requires ~21k multiply‑adds per item, still far faster than tree ensembles or kNN. If dev accuracy does not exceed label‑linear (0.9613) by at least 0.5 %, or if inference cost exceeds label‑mlp16 (8685 Ir/item), the family does not justify its complexity.
```
```python
"""Polynomial logistic regression (degree 2) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread. Uses only stdlib + numpy.
Expands each byte (after per‑byte z‑scoring) into 1 + d + d(d+1)/2 features,
then trains a multinomial logistic regression with Adam (lr=3e-3, L2=1e-4,
batch=64, epochs=min(100, ceil(600000/n))). Hyper‑parameters (lr, L2) are chosen
on the last 20% of train.in (file order), then retrained on all data.
Weights file format (little‑endian):
  int32 magic=0x504F4C32, int32 d, int32 C,
  float32 mean[d], float32 std[d],
  float32 weights[(1+d+d(d+1)/2) * C]  (row‑major: feature‑major, then class)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
LR_GRID = [0.01, 0.03, 0.1]
L2_GRID = [1e-4, 1e-3, 1e-2]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def poly_expand(Z):
    """Z: (n, d) normalized features -> (n, 1 + d + d(d+1)/2)"""
    n, d = Z.shape
    D = 1 + d + d * (d + 1) // 2
    Phi = np.empty((n, D), dtype=np.float32)
    Phi[:, 0] = 1.0
    Phi[:, 1:1 + d] = Z
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = Z[:, i] * Z[:, j]
            col += 1
    return Phi


def softmax_grad(logits, y):
    """logits: (batch, C), y: (batch,) -> gradient of CE w.r.t logits"""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_one(X_tr, y_tr, X_val, y_val, lr, l2, rng, epochs):
    n, D = X_tr.shape
    C = int(y_tr.max()) + 1
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    best_val = -1.0
    best_W = None
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits = Xb @ W
            grad = Xb.T @ softmax_grad(logits, yb) + l2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # validation
        val_logits = X_val @ W
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_W = W.copy()
    return best_W, best_val


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # per‑byte z‑scoring (ordinal only, all are ordinal)
    mean = X_bytes.mean(axis=0).astype(np.float64)
    std = X_bytes.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    Z = ((X_bytes - mean) / std).astype(np.float32)

    # polynomial expansion
    Phi = poly_expand(Z)
    D = Phi.shape[1]

    # split: last 20% for tuning
    split = int(0.8 * n)
    idx = np.arange(n)
    tr_idx, val_idx = idx[:split], idx[split:]
    Phi_tr, y_tr = Phi[tr_idx], y[tr_idx]
    Phi_val, y_val = Phi[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_W = None
    best_score = -1.0
    best_lr = LR_GRID[0]
    best_l2 = L2_GRID[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(tr_idx)))

    for lr in LR_GRID:
        for l2 in L2_GRID:
            W, val_acc = train_one(Phi_tr, y_tr, Phi_val, y_val, lr, l2, rng, epochs)
            if val_acc > best_score:
                best_score = val_acc
                best_W = W
                best_lr = lr
                best_l2 = l2

    # retrain on all data with best hyper‑parameters
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    W_final, _ = train_one(Phi, y, Phi[:1], y[:1], best_lr, best_l2, rng, epochs_full)

    # write weights file
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<3i", magic, d, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Polynomial logistic regression (degree 2) inference.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (pixels). Output: one byte class index. */
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
    if (nw < 12 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d=%d C=%d\n", d, C); return 3; }
    int D = 1 + d + d * (d + 1) / 2;  // 2145
    long expected = 12L + 4L * d * 2 + 4L * D * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *std  = mean + d;
    const float *W    = std + d;  // size D * C, row-major: feature-major then class

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *phi   = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize
        for (int j = 0; j < d; j++) {
            float s = std[j];
            xnorm[j] = s > 0.0f ? (xi[j] - mean[j]) / s : 0.0f;
        }

        // polynomial features: phi[0]=1, phi[1..d]=xnorm, phi[1+d..]=quadratic
        phi[0] = 1.0f;
        memcpy(phi + 1, xnorm, d * sizeof(float));
        int col = 1 + d;
        for (int u = 0; u < d; u++) {
            for (int v = u; v < d; v++) {
                phi[col++] = xnorm[u] * xnorm[v];
            }
        }

        // scores = phi^T W
        for (int c = 0; c < C; c++) score[c] = 0.0f;
        for (int f = 0; f < D; f++) {
            float ph = phi[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) score[c] += ph * wf[c];
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```