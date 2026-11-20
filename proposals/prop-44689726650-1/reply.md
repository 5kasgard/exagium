```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).  
Why: The 8×8 optdigits pixels have strong local correlations; pairwise products capture stroke‑like interactions without a hidden layer. With 64 ordinal features, the polynomial expansion (bias + 64 linear + 2080 quadratic = 2145 features) is still small enough for exact multinomial logistic regression. Regularized training on this space should beat label-linear (0.9613) and approach label-mlp16 (0.9770) at lower inference cost than an MLP.  
Failure: If dev accuracy ≤ 0.9613 or instructions/item exceed label-mlp16’s 8685, the family does not meet its goal.
```

```python
"""label-poly2: multinomial logistic regression on degree-2 polynomial features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: uint32 magic 0x504F4C32, int32 d, C, F; float32 mu[d], sd[d]; float32 W[F][C] row-major.
"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2_GRID = [1e-4, 1e-3, 1e-2]   # tuned on last 20% of train
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


def poly_features(Z):
    """Z: (n, d) normalized float32 -> (n, F) with F = 1 + d + d*(d+1)//2"""
    n, d = Z.shape
    F = 1 + d + d * (d + 1) // 2
    Phi = np.empty((n, F), dtype=np.float32)
    Phi[:, 0] = 1.0
    Phi[:, 1:1 + d] = Z
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = Z[:, i] * Z[:, j]
            col += 1
    return Phi


def softmax_grad(logits, y):
    """logits (n, C), y (n,) -> gradient (n, C)"""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_eval(Phi_tr, y_tr, Phi_va, y_va, C, L2, rng):
    n, F = Phi_tr.shape
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    m = np.zeros_like(W); v = np.zeros_like(W)
    t = 0
    best_acc = -1.0
    best_W = W.copy()
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = Phi_tr[idx], y_tr[idx]
            logits = Xb @ W
            grad = Xb.T @ softmax_grad(logits, yb) + L2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # validation accuracy
        va_logits = Phi_va @ W
        va_pred = va_logits.argmax(axis=1)
        acc = (va_pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_W = W.copy()
    return best_W, best_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning (file order)
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    # normalization on training split
    mu = X_tr.mean(axis=0).astype(np.float64)
    sd = X_tr.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z_tr = ((X_tr - mu) / sd).astype(np.float32)
    Z_va = ((X_va - mu) / sd).astype(np.float32)

    Phi_tr = poly_features(Z_tr)
    Phi_va = poly_features(Z_va)
    F = Phi_tr.shape[1]

    rng = np.random.default_rng(SEED)
    best_L2 = L2_GRID[0]
    best_W = None
    best_acc = -1.0
    for L2 in L2_GRID:
        W, acc = train_eval(Phi_tr, y_tr, Phi_va, y_va, C, L2, rng)
        if acc > best_acc:
            best_acc = acc
            best_L2 = L2
            best_W = W

    # retrain on full training set with best L2
    mu_full = X.mean(axis=0).astype(np.float64)
    sd_full = X.std(axis=0).astype(np.float64)
    sd_full[sd_full == 0] = 1.0
    Z_full = ((X - mu_full) / sd_full).astype(np.float32)
    Phi_full = poly_features(Z_full)
    # reinitialize and train to convergence (fixed epochs)
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    m = np.zeros_like(W); v = np.zeros_like(W)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = Phi_full[idx], y[idx]
            logits = Xb @ W
            grad = Xb.T @ softmax_grad(logits, yb) + best_L2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x504F4C32  # "POL2"
    header = struct.pack("<4i", magic, d, C, F)
    mu_bytes = mu.astype("<f4").tobytes()
    sd_bytes = sd_full.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()  # row-major (F, C)
    open(sys.argv[4], "wb").write(header + mu_bytes + sd_bytes + W_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte class index.
 * Weights format: magic 0x504F4C32, int32 d, C, F; float32 mu[d], sd[d]; float32 W[F][C] row-major.
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || F != 1 + d + d * (d + 1) / 2) {
        fprintf(stderr, "bad weights dimensions\n"); return 3; }
    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *W = sd + d;  // size F * C
    long expected = 16 + (long)d * 2 * 4 + (long)F * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));
    float *phi = malloc(F * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // normalize
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) / sd[j];
        // polynomial features
        int col = 0;
        phi[col++] = 1.0f;
        for (int j = 0; j < d; j++) phi[col++] = z[j];
        for (int u = 0; u < d; u++) {
            for (int v = u; v < d; v++) {
                phi[col++] = z[u] * z[v];
            }
        }
        // scores = phi^T W
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int f = 0; f < F; f++) {
            float ph = phi[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += ph * wf[c];
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```