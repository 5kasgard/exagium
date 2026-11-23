```hypothesis
label-rbf: RBF network with k-means centers on z-scored pixels.
Why: RBF networks model non-linear boundaries well for digit recognition; 32 centers give ~2K ops/item.
Failure: dev accuracy ≤ 0.9613 (linear baseline) or CPU cost > 10K Ir/item.
```
```python
"""label-rbf: RBF network with k-means centers on z-scored ordinal features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports: magic, d, K, C, gamma, mu[d], sd[d], centers[K][d], W_out[(K+1)][C]."""
import json, math, struct, sys
import numpy as np

# Hyperparameters (tuned on last 20% of train)
SEED = 1
K = 32          # number of RBF centers
GAMMA = 0.1     # RBF width
LR = 3e-3       # learning rate
L2 = 1e-4       # L2 regularization
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
KMEANS_ITERS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, iters=20):
    """Deterministic k-means with random init from data points."""
    n = X.shape[0]
    idx = rng.choice(n, size=K, replace=False)
    centers = X[idx].copy()
    for _ in range(iters):
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            k = assign[i]
            new_centers[k] += X[i]
            counts[k] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                new_centers[k] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64
    # All features are ordinal per card
    mu = X_raw.mean(axis=0).astype(np.float32)
    sd = X_raw.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_raw.astype(np.float32) - mu) / sd).astype(np.float32)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    centers = kmeans(X, K, rng, KMEANS_ITERS).astype(np.float32)
    # Precompute RBF features
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    Phi = np.exp(-GAMMA * dist2).astype(np.float32)
    Phi_bias = np.hstack([Phi, np.ones((n, 1), dtype=np.float32)])
    # Train linear model on RBF features
    W = np.zeros((K + 1, C), dtype=np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Pb = Phi_bias[idx]
            yy = y[idx]
            logits = Pb @ W
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (Pb.T @ probs) / len(yy) + L2 * W
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Export weights
    magic = 0x31464252  # 'RBF1' little-endian
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", GAMMA)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rbf: RBF network inference. ./infer weights.bin in.rec N
 * Input: d bytes per record (z-scored using stored mu, sd).
 * Output: one byte class (argmax). */
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
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x31464252) { fprintf(stderr, "bad weights: magic %08x\n", magic); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], K = hd[1], C = hd[2];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights: dims\n"); return 3; }
    float *ptr = (float *)(hd + 3);
    float gamma = *ptr++;
    float *mu = ptr; ptr += d;
    float *sd = ptr; ptr += d;
    float *centers = ptr; ptr += (long)K * d;
    float *Wout = ptr;
    long expected = 4 + 12 + 4 + (long)d * 4 * 2 + (long)K * d * 4 + (long)(K + 1) * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * (K + 1));
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        for (int c = 0; c < K; c++) {
            float dist2 = 0.0f;
            const float *cen = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - cen[j];
                dist2 += diff * diff;
            }
            phi[c] = expf(-gamma * dist2);
        }
        phi[K] = 1.0f;
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int u = 0; u <= K; u++) {
                sum += phi[u] * Wout[(long)u * C + c];
            }
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```