```hypothesis
Family: label-fm (Factorization Machine with pairwise interactions). 
Adult has 14 mixed ordinal/categorical features; a degree-2 FM captures feature crosses that linear models miss, while staying much smaller and faster than GBDT/MLP. With k=4 latent dimensions, the model adds ~18k parameters (vs linear's ~3k) and inference costs ~14*k FLOPs per item. We expect dev accuracy >0.855 (beating label-linear's 0.8524) at ~2x linear's instruction count. If dev accuracy ≤0.8524 or instructions/item >1500, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal -> z-scored (1 dim), categorical -> one-hot over training values.
FM: logit = w0 + Σ w_i z_i + Σ_{i<j} <v_i, v_j> z_i z_j.
Trained with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Hyperparameter k (latent dim) chosen on last 20% of train (tuning split).
Exported as per-input-byte lookup tables: int32 magic 0x464D3132, n_in, k, C=2;
float32 w0, linear_contrib[n_in][256], V_contrib[n_in][256][k], norm_sq_contrib[n_in][256]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [4, 8]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encodings(X, types):
    """Return (dim_start, dim_end, mapping) where mapping[j][byte] = list of (dim, z_value)."""
    n, d_in = X.shape
    dim_start = [0] * d_in
    dim_end = [0] * d_in
    mapping = [{} for _ in range(d_in)]
    dim = 0
    for j, t in enumerate(types):
        dim_start[j] = dim
        v = X[:, j]
        if t == "ordinal":
            mu = v.mean()
            sd = v.std() or 1.0
            for byte in range(256):
                if byte == 255:
                    z = 0.0
                else:
                    z = (byte - mu) / sd
                mapping[j][byte] = [(dim, z)]
            dim += 1
        else:
            vals = np.unique(v)
            cat_to_idx = {c: i for i, c in enumerate(vals)}
            for byte in range(256):
                if byte in cat_to_idx:
                    mapping[j][byte] = [(dim + cat_to_idx[byte], 1.0)]
                else:
                    mapping[j][byte] = []
            dim += len(vals)
        dim_end[j] = dim
    return dim_start, dim_end, mapping, dim


def train_fm(X, y, mapping, dim, k, rng, n_train):
    w0 = 0.0
    w = np.zeros(dim, np.float32)
    v = (rng.normal(0, 0.01, (dim, k))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    for _ in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            z_batch = X[idx]
            y_batch = y[idx]
            bs = len(idx)
            # forward
            linear = np.full(bs, w0, np.float32)
            S = np.zeros((bs, k), np.float32)
            norm_sq_sum = np.zeros(bs, np.float32)
            for j in range(z_batch.shape[1]):
                bytes_j = z_batch[:, j]
                for byte in range(256):
                    mask = (bytes_j == byte)
                    if not np.any(mask):
                        continue
                    for (m, z_val) in mapping[j][byte]:
                        linear[mask] += w[m] * z_val
                        S[mask] += v[m] * z_val
                        norm_sq_sum[mask] += np.dot(v[m], v[m]) * (z_val * z_val)
            interaction = 0.5 * (np.sum(S * S, axis=1) - norm_sq_sum)
            logit = linear + interaction
            # sigmoid gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - y_batch).astype(np.float32)  # bs
            # backward
            g_w0 = grad.mean()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for j in range(z_batch.shape[1]):
                bytes_j = z_batch[:, j]
                for byte in range(256):
                    mask = (bytes_j == byte)
                    if not np.any(mask):
                        continue
                    g = grad[mask]
                    for (m, z_val) in mapping[j][byte]:
                        g_w[m] += np.dot(g, np.full_like(g, z_val)) / bs
                        g_v[m] += np.outer(g, v[m]).sum(axis=0) * z_val / bs
                        # derivative of interaction term w.r.t v[m]
                        # d/d v[m] 0.5*(||S||^2 - sum ||v_i||^2 z_i^2) = S * z_val - v[m] * z_val^2
                        g_v[m] += (np.dot(g, S[mask] * z_val - v[m] * (z_val * z_val))) / bs
            # L2
            g_w += L2 * w
            g_v += L2 * v
            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            lr_t = LR * (np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
            w0 -= lr_t * m_w0 / (np.sqrt(v_w0) + 1e-8)
            w -= lr_t * m_w / (np.sqrt(v_w) + 1e-8)
            v -= lr_t * m_v / (np.sqrt(v_v) + 1e-8)
    return w0, w, v


def evaluate_fm(X, y, mapping, w0, w, v, k):
    n = X.shape[0]
    linear = np.full(n, w0, np.float32)
    S = np.zeros((n, k), np.float32)
    norm_sq_sum = np.zeros(n, np.float32)
    for j in range(X.shape[1]):
        bytes_j = X[:, j]
        for byte in range(256):
            mask = (bytes_j == byte)
            if not np.any(mask):
                continue
            for (m, z_val) in mapping[j][byte]:
                linear[mask] += w[m] * z_val
                S[mask] += v[m] * z_val
                norm_sq_sum[mask] += np.dot(v[m], v[m]) * (z_val * z_val)
    interaction = 0.5 * (np.sum(S * S, axis=1) - norm_sq_sum)
    logit = linear + interaction
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]

    # build encodings on full training set (as per card: choose settings on tuning split, then train on all)
    dim_start, dim_end, mapping, dim = build_encodings(X, types)

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + k)
        w0, w, v = train_fm(X_tr, y_tr, mapping, dim, k, rng_k, len(idx_tr))
        acc = evaluate_fm(X_va, y_va, mapping, w0, w, v, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # retrain on all data with best_k
    rng_final = np.random.default_rng(SEED + best_k)
    w0, w, v = train_fm(X, y, mapping, dim, best_k, rng_final, n)

    # precompute per-byte lookup tables
    n_in = card["input"]["bytes"]
    linear_contrib = np.zeros((n_in, 256), np.float32)
    V_contrib = np.zeros((n_in, 256, best_k), np.float32)
    norm_sq_contrib = np.zeros((n_in, 256), np.float32)
    for j in range(n_in):
        for byte in range(256):
            lin = 0.0
            vec = np.zeros(best_k, np.float32)
            ns = 0.0
            for (m, z_val) in mapping[j][byte]:
                lin += w[m] * z_val
                vec += v[m] * z_val
                ns += np.dot(v[m], v[m]) * (z_val * z_val)
            linear_contrib[j, byte] = lin
            V_contrib[j, byte] = vec
            norm_sq_contrib[j, byte] = ns

    # write weights.bin
    MAGIC = 0x464D3132  # "FM12"
    C = card["classes"]
    out = struct.pack("<4i", MAGIC, n_in, best_k, C)
    out += struct.pack("<f", w0)
    out += linear_contrib.astype("<f4").tobytes()
    out += V_contrib.astype("<f4").tobytes()
    out += norm_sq_contrib.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks with fixed-size byte items.
 * ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   int32 magic=0x464D3132, n_in, k, C
 *   float32 w0
 *   float32 linear_contrib[n_in][256]
 *   float32 V_contrib[n_in][256][k]
 *   float32 norm_sq_contrib[n_in][256]
 * Input record: 4-byte LE length (must equal n_in), then n_in bytes.
 * Output record: 4-byte LE length 1, then class byte (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_in = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || n_in <= 0 || k <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    size_t expected = 16
        + sizeof(float)                      // w0
        + n_in * 256 * sizeof(float)         // linear_contrib
        + n_in * 256 * k * sizeof(float)     // V_contrib
        + n_in * 256 * sizeof(float);        // norm_sq_contrib
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float w0 = *p++;
    const float *linear_contrib = p; p += n_in * 256;
    const float *V_contrib = p; p += n_in * 256 * k;
    const float *norm_sq_contrib = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != n_in) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, n_in); return 3; }

        float linear = w0;
        float S[16];  // k <= 16 (we only use 4 or 8)
        float norm_sq_sum = 0.0f;
        for (int u = 0; u < k; u++) S[u] = 0.0f;

        for (int j = 0; j < n_in; j++) {
            unsigned char byte = xi[j];
            linear += linear_contrib[j * 256 + byte];
            const float *Vj = V_contrib + (j * 256 + byte) * k;
            for (int u = 0; u < k; u++) S[u] += Vj[u];
            norm_sq_sum += norm_sq_contrib[j * 256 + byte];
        }

        float dot = 0.0f;
        for (int u = 0; u < k; u++) dot += S[u] * S[u];
        float interaction = 0.5f * (dot - norm_sq_sum);
        float logit = linear + interaction;

        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```