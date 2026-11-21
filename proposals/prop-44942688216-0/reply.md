```hypothesis
Family: label-fm (Factorization Machine with binary output). Captures pairwise feature interactions via low-rank factors, which helps on tabular data with mixed ordinal/categorical columns where interactions matter (e.g., education × occupation). Uses the same per-column encoding as label-linear (ordinal standardized, categorical one-hot) but adds a factorized interaction term. Expected to beat label-linear accuracy (0.8524) with modest cost increase; if dev accuracy ≤ 0.8524 or Ir/item > 1500, the model does not justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored, categorical as one-hot over training values (255=unseen→zeros).
Learns w0, w[D], V[D,k] with logistic loss, Adam (lr=3e-3, L2=1e-4), batch=64, epochs=min(100, ceil(600000/n)).
Chooses latent dimension k on last 20% of train.in (tuning split), then retrains on all data.
Weights format (little-endian):
  int32 magic=0x464D0001, d=14, D, k, C=2
  int32 feature_dim[14]
  for each column: float32 feat_vec[256 * feature_dim[j]] (row-major)
  float32 w0, w[D], V[D*k] (row-major)
"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [4, 8, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(card, X):
    """Return list of (feature_dim, feat_vec[256, feature_dim]) per column."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            vec = ((np.arange(256) - mu) / sd).astype(np.float32)[:, None]  # (256,1)
            enc.append((1, vec))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append((len(vals), E))
    return enc

def encode_all(X, enc):
    """Concatenate per-column feature vectors into big matrix (n, D)."""
    parts = [E[X[:, j]] for j, (_, E) in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)

def train_fm(Z, y, k, epochs, rng):
    n, D = Z.shape
    w0 = np.float32(0.0)
    w = np.zeros(D, np.float32)
    V = (rng.normal(0, 0.01, (D, k))).astype(np.float32)
    params = [w0, w, V]
    # Adam state: for w0, w, V
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)
            # forward
            linear = w0 + z @ w
            S = z @ V                     # (bsz, k)
            sum_sq = np.sum(S * S, axis=1)
            sq_sum = np.sum((z * z) @ (V * V), axis=1)
            interaction = 0.5 * (sum_sq - sq_sum)
            score = linear + interaction  # (bsz,)
            # logistic gradient
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yy).astype(np.float32) / bsz  # (bsz,)
            # gradients
            gw0 = g.sum()
            gw = z.T @ g + L2 * w
            # d interaction / dV = z.T @ (g[:,None] * S) - (z*z).T @ (g[:,None] * V)
            gS = g[:, None] * S
            gV = g[:, None] * V
            gV = (z * z).T @ gV
            gV = z.T @ gS - gV + L2 * V
            grads = [np.array(gw0, np.float32), gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            w0, w, V = params[0], params[1], params[2]
    return w0, w, V

def accuracy(Z, y, w0, w, V):
    linear = w0 + Z @ w
    S = Z @ V
    sum_sq = np.sum(S * S, axis=1)
    sq_sum = np.sum((Z * Z) @ (V * V), axis=1)
    score = linear + 0.5 * (sum_sq - sq_sum)
    pred = (score > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # encoding on all training data (for final model) but we need tuning split indices
    split = int(0.8 * n)
    idx_all = np.arange(n)
    idx_train, idx_val = idx_all[:split], idx_all[split:]
    # build encoding using only train split? The card says tuning split is last 20% of train.in in file order.
    # We must choose hyperparameters on tuning split, but encoding should be based on all training data? 
    # The example label-linear uses all training data for encoding. We'll follow that: encoding on all X.
    enc = build_encoding(card, X)
    D = sum(d for d, _ in enc)
    Z_all = encode_all(X, enc)
    # hyperparameter search on tuning split
    best_k, best_acc = K_CHOICES[0], -1.0
    for k in K_CHOICES:
        rng = np.random.default_rng(SEED + k)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
        w0, w, V = train_fm(Z_all[idx_train], y[idx_train], k, epochs, rng)
        acc = accuracy(Z_all[idx_val], y[idx_val], w0, w, V)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # retrain on all data with best_k
    rng = np.random.default_rng(SEED + best_k)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, w, V = train_fm(Z_all, y, best_k, epochs, rng)
    # write weights
    feature_dims = np.array([d for d, _ in enc], dtype=np.int32)
    feat_vecs = [E for _, E in enc]
    out = bytearray()
    out += struct.pack("<5i", 0x464D0001, X.shape[1], D, best_k, 2)
    out += feature_dims.tobytes()
    for E in feat_vecs:
        out += E.astype("<f4").tobytes()
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: see train.py. Input records: 14 bytes each. Output: one byte per record (0 or 1). */
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
    if (nw < 20 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], k = hd[3], C = hd[4];
    if (d != 14 || C != 2 || k <= 0 || D <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    int32_t *feature_dim = (int32_t *)(hd + 5);
    const float *feat_vecs = (const float *)(feature_dim + d);
    long expected = 20L + 4L * d;
    for (int j = 0; j < d; j++) expected += 4L * 256 * feature_dim[j];
    expected += 4L * (1 + D + (long)D * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *w0 = feat_vecs;
    for (int j = 0; j < d; j++) w0 = (const float *)((const char *)w0 + 4L * 256 * feature_dim[j]);
    const float *w = w0 + 1;
    const float *V = w + D;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xvec = (float *)malloc(sizeof(float) * D);
    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if (klen != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }
        memset(xvec, 0, sizeof(float) * D);
        int offset = 0;
        const float *fv = feat_vecs;
        for (int j = 0; j < d; j++) {
            int fd = feature_dim[j];
            const float *col_vec = fv + 256 * xi[j] * fd;
            for (int f = 0; f < fd; f++) xvec[offset + f] = col_vec[f];
            offset += fd;
            fv += 256 * fd;
        }
        float linear = w0[0];
        for (int idx = 0; idx < D; idx++) linear += xvec[idx] * w[idx];
        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            float sum_Vx = 0.0f, sum_Vx_sq = 0.0f;
            for (int idx = 0; idx < D; idx++) {
                float v = V[(long)idx * k + f];
                float xi = xvec[idx];
                sum_Vx += v * xi;
                sum_Vx_sq += v * v * xi * xi;
            }
            interaction += 0.5f * (sum_Vx * sum_Vx - sum_Vx_sq);
        }
        float score = linear + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(xvec);
    return 0;
}
```