```hypothesis
Family: label-rbf (Radial Basis Function network with k-means centers). 
Why: Optdigits pixels are ordinal 0–16; RBF on normalized pixels captures local similarities better than a linear model. 
A small number of centers (M≈40) keeps inference fast (≈7k Ir/item) while adding non-linearity to beat label-linear (0.9613). 
Failure: If dev accuracy ≤ 0.9613 or Ir/item ≥ 10000, the family does not improve on label-linear.
```

```python
"""label-rbf: RBF network with k-means centers + linear readout.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the tuning split (last 20% of train.in)
M_GRID = [20, 30, 40, 50, 60]
GAMMA_GRID = [0.01, 0.1, 1.0, 10.0]

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
KM_ITERS = 100          # k-means iterations
KM_SEED = SEED + 123    # separate seed for k-means


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def kmeans(X, M, rng):
    """Lloyd's algorithm, returns centers (M, d)."""
    n, d = X.shape
    idx = rng.choice(n, M, replace=False)
    centers = X[idx].copy()
    for _ in range(KM_ITERS):
        # Assign
        diff = X[:, None, :] - centers[None, :, :]   # (n, M, d)
        dist2 = (diff * diff).sum(axis=2)            # (n, M)
        assign = dist2.argmin(axis=1)                # (n,)
        # Update
        new_centers = np.zeros_like(centers)
        counts = np.zeros(M, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for c in range(M):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
            else:
                # Re-initialize empty cluster
                new_centers[c] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def rbf_features(X, centers, gamma):
    """X: (n, d), centers: (M, d) -> (n, M)"""
    diff = X[:, None, :] - centers[None, :, :]       # (n, M, d)
    dist2 = (diff * diff).sum(axis=2)                # (n, M)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_linear(Z, y, C, rng):
    """Adam on softmax cross-entropy, returns W (M, C), b (C)."""
    n, M = Z.shape
    W = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)   # (batch, C)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def evaluate(X, y, centers, gamma, W, b):
    Z = rbf_features(X, centers, gamma)
    logits = Z @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Split: last 20% is tuning (file order)
    n = X.shape[0]
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Normalize on training portion
    X_tr_n, mean, std = normalize(X_tr.astype(np.float64))
    X_tune_n = normalize(X_tune.astype(np.float64), mean, std)[0]

    rng = np.random.default_rng(SEED)
    rng_km = np.random.default_rng(KM_SEED)

    best_acc = -1.0
    best_M = M_GRID[0]
    best_gamma = GAMMA_GRID[0]
    best_centers = None
    best_W = None
    best_b = None

    for M in M_GRID:
        centers = kmeans(X_tr_n, M, rng_km)
        for gamma in GAMMA_GRID:
            Z_tr = rbf_features(X_tr_n, centers, gamma)
            W, b = train_linear(Z_tr, y_tr, C, rng)
            Z_tune = rbf_features(X_tune_n, centers, gamma)
            logits = Z_tune @ W + b
            pred = logits.argmax(axis=1)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_M = M
                best_gamma = gamma
                best_centers = centers.copy()
                best_W = W.copy()
                best_b = b.copy()

    # Retrain on full dataset with best hyperparameters
    X_full_n, mean_full, std_full = normalize(X.astype(np.float64))
    centers_full = kmeans(X_full_n, best_M, rng_km)
    Z_full = rbf_features(X_full_n, centers_full, best_gamma)
    W_full, b_full = train_linear(Z_full, y, C, rng)

    # Write weights.bin
    # Magic: 0x52424631 ('RBF1' little-endian)
    # Layout: int32 magic, M, d, C; float32 gamma; float32 mean[d], std[d];
    #         float32 centers[M*d]; float32 W[M*C]; float32 b[C]
    magic = 0x52424631
    out = struct.pack("<5i f", magic, best_M, d, C, best_gamma)
    out += mean_full.astype("<f4").tobytes()
    out += std_full.astype("<f4").tobytes()
    out += best_centers.astype("<f4").tobytes()
    out += W_full.astype("<f4").tobytes()
    out += b_full.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format (little-endian):
 *   int32 magic (0x52424631), M, d, C; float32 gamma;
 *   float32 mean[d], std[d]; float32 centers[M*d]; float32 W[M*C]; float32 b[C] */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int M = hd[1], d = hd[2], C = hd[3];
    if (M <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float *fp = (float *)(hd + 4);
    float gamma = *fp++;
    float *mean = fp; fp += d;
    float *std  = fp; fp += d;
    float *centers = fp; fp += (long)M * d;
    float *W = fp; fp += (long)M * C;
    float *b = fp;
    long expected = 20 + 4L * (1 + 2*d + (long)M*d + (long)M*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *rbf = malloc(sizeof(float) * M);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int m = 0; m < M; m++) {
            float dist2 = 0.0f;
            const float *c = centers + (long)m * d;
            for (int j = 0; j < d; j++) {
                float diff = xnorm[j] - c[j];
                dist2 += diff * diff;
            }
            rbf[m] = expf(-gamma * dist2);
        }
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int m = 0; m < M; m++) {
            float rm = rbf[m];
            const float *wm = W + (long)m * C;
            for (int c = 0; c < C; c++) logits[c] += rm * wm[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```