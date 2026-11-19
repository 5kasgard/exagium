```hypothesis
Family: label-rbf (RBF network with class centroids). The 8x8 digit pixels are ordinal; per-pixel z-scoring gives a Euclidean space where class centroids are well-separated. An RBF layer with one Gaussian per class centroid captures non-linear boundaries better than linear, while the second linear layer combines them. Inference costs ~700 FLOPs/item (640 for distances, 10 exps, 100 for linear). Beats label-linear (0.961) if centroids are discriminative; fails if classes overlap heavily in this space (dev accuracy ≤ 0.96).
```

```python
import json, math, struct, sys
import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
GAMMA_CANDIDATES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode(X, mu, sd):
    return (X.astype(np.float64) - mu) / sd

def compute_centroids(Z, y, C):
    d = Z.shape[1]
    cents = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            cents[c] = Z[mask].mean(axis=0)
    return cents

def rbf_features(Z, cents, gamma):
    Z2 = np.sum(Z * Z, axis=1, keepdims=True)
    C2 = np.sum(cents * cents, axis=1)
    dist = Z2 + C2 - 2 * Z @ cents.T
    np.maximum(dist, 0, out=dist)
    return np.exp(-gamma * dist).astype(np.float32)

def train_linear(Phi, y, C, rng):
    n, k = Phi.shape
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Phi[idx], y[idx]
            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    n_tune = n // 5
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)
    rng = np.random.default_rng(SEED)
    mu = X[train_idx].mean(axis=0).astype(np.float64)
    sd = X[train_idx].std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z_train = encode(X[train_idx], mu, sd)
    Z_tune = encode(X[tune_idx], mu, sd)
    y_train = y[train_idx]
    y_tune = y[tune_idx]
    cents = compute_centroids(Z_train, y_train, C)
    best_gamma, best_acc = GAMMA_CANDIDATES[0], -1.0
    for gamma in GAMMA_CANDIDATES:
        Phi_train = rbf_features(Z_train, cents, gamma)
        W, b = train_linear(Phi_train, y_train, C, rng)
        Phi_tune = rbf_features(Z_tune, cents, gamma)
        scores = Phi_tune @ W + b
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_gamma = acc, gamma
    mu_all = X.mean(axis=0).astype(np.float64)
    sd_all = X.std(axis=0).astype(np.float64)
    sd_all[sd_all == 0] = 1.0
    Z_all = encode(X, mu_all, sd_all)
    cents_all = compute_centroids(Z_all, y, C)
    Phi_all = rbf_features(Z_all, cents_all, best_gamma)
    W_final, b_final = train_linear(Phi_all, y, C, rng)
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<iii", 0x52424631, d, C))
        f.write(struct.pack("<f", best_gamma))
        f.write(mu_all.astype(np.float32).tobytes())
        f.write(sd_all.astype(np.float32).tobytes())
        f.write(cents_all.astype(np.float32).tobytes())
        f.write(W_final.astype(np.float32).tobytes())
        f.write(b_final.astype(np.float32).tobytes())

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
    if (hd[0] != 0x52424631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float *ptr = (float *)(hd + 3);
    float gamma = ptr[0];
    float *mu = ptr + 1;
    float *sd = mu + d;
    float *cents = sd + d;
    float *W2 = cents + (long)C * d;
    float *b2 = W2 + (long)C * C;
    long expected = 16 + 4L * (1 + 2*d + (long)C*d + (long)C*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *rbf = malloc(sizeof(float) * C);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        for (int c = 0; c < C; c++) {
            float dist = 0.0f;
            const float *cent = cents + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - cent[j];
                dist += diff * diff;
            }
            rbf[c] = expf(-gamma * dist);
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int k = 0; k < C; k++) {
                sum += rbf[k] * W2[(long)k * C + c];
            }
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```