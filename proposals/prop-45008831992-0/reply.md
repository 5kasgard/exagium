```hypothesis
label-fm: Factorization Machine with class-specific latent vectors. Captures pairwise feature interactions crucial for tabular data like Adult, while remaining lightweight. Uses the same ordinal/categorical encoding as label-linear but adds a low-rank interaction term (latent dim k). Tuning k on the last 20% of train.in should yield >0.8524 dev accuracy. If dev accuracy ≤ label-linear, interactions are not beneficial or training is insufficient.
```

```python
"""label-fm: Factorization Machine for tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as z-scored, categorical as one-hot (unseen=255 -> zero).
Trains FM with class-specific V (latent dim k) using Adam.
Selects k on last 20% of train.in (file order), then retrains on all.
Exports per-byte lookup tables: magic 0x464D3132, d, k, C;
float32 w0[C], lin[d][256][C], lat[d][256][k][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8, 16]  # latent dimensions to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_features(card, X):
    """Return Z (n x D), and metadata for lookup table construction."""
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    enc = []          # list of (type, param) for each original feature
    Z_cols = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            scaled = (np.arange(256, dtype=np.float64) - mu) / sd
            # Z column is the scaled value for each sample
            Z_cols.append(scaled[X[:, j]].reshape(-1, 1).astype(np.float32))
            enc.append(("ordinal", (mu, sd)))
        else:
            vals = np.unique(X[:, j])
            val_to_idx = {val: i for i, val in enumerate(vals)}
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            Z_cols.append(E[X[:, j]].astype(np.float32))
            enc.append(("categorical", (vals, val_to_idx)))
    Z = np.concatenate(Z_cols, axis=1)
    return Z, enc

def fm_forward(Z, w0, w, V, C, k):
    """Compute scores (n x C) for FM."""
    n = Z.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    scores += w0
    scores += Z @ w
    # interaction term
    for c in range(C):
        A = Z @ V[:, :, c]          # n x k
        B = (Z * Z) @ (V[:, :, c] * V[:, :, c])  # n x k
        scores[:, c] += 0.5 * np.sum(A * A - B, axis=1)
    return scores

def fm_backward(Z, grad_scores, w, V, C, k, L2):
    """Compute gradients for w0, w, V."""
    n = Z.shape[0]
    grad_w0 = grad_scores.sum(axis=0)                     # C
    grad_w = Z.T @ grad_scores                            # D x C
    grad_V = np.zeros_like(V)                             # D x k x C
    Z2 = Z * Z
    for c in range(C):
        g = grad_scores[:, c]                             # n
        for f in range(k):
            V_fc = V[:, f, c]                             # D
            S = Z @ V_fc                                  # n
            coeff = S * g                                 # n
            grad_V[:, f, c] = Z.T @ coeff - Z2.T @ (V_fc * g)
    grad_w += L2 * w
    grad_V += L2 * V
    return grad_w0, grad_w, grad_V

def train_fm(Z, y, C, k, seed, n_samples=None):
    """Train FM on (Z,y) with Adam. Returns w0, w, V."""
    n, D = Z.shape
    rng = np.random.default_rng(seed)
    # init
    w0 = np.zeros(C, dtype=np.float32)
    w = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    V = (rng.uniform(-1, 1, (D, k, C)) * math.sqrt(6.0 / (D * k))).astype(np.float32)
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n)) if n_samples is None else math.ceil(n_samples / n)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            scores = fm_forward(z, w0, w, V, C, k)
            # softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = (probs / len(yy)).astype(np.float32)
            g_w0, g_w, g_V = fm_backward(z, grad_scores, w, V, C, k, L2)
            grads = [g_w0, g_w, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params[0], params[1], params[2]

def evaluate(Z, y, w0, w, V, C, k):
    scores = fm_forward(Z, w0, w, V, C, k)
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # Encode using all train data (for consistent scaling/category mapping)
    Z_all, enc = encode_features(card, X)
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    Z_train, Z_val = Z_all[:split], Z_all[split:]
    y_train, y_val = y[:split], y[split:]

    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        seed_k = SEED + k * 1000
        w0, w, V = train_fm(Z_train, y_train, C, k, seed_k)
        acc = evaluate(Z_val, y_val, w0, w, V, C, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # Retrain on all data with best_k
    w0, w, V = train_fm(Z_all, y, C, best_k, SEED)

    # Build lookup tables: lin[d][256][C], lat[d][256][k][C]
    lin = np.zeros((d, 256, C), dtype=np.float32)
    lat = np.zeros((d, 256, best_k, C), dtype=np.float32)

    # Map original feature to encoded column indices
    col_idx = 0
    for j, (typ, param) in enumerate(enc):
        if typ == "ordinal":
            mu, sd = param
            scaled = (np.arange(256, dtype=np.float32) - mu) / sd
            for c in range(C):
                lin[j, :, c] = w[col_idx, c] * scaled
                lat[j, :, :, c] = V[col_idx, :, c].reshape(1, best_k) * scaled.reshape(256, 1)
            col_idx += 1
        else:
            vals, val_to_idx = param
            for v in range(256):
                if v in val_to_idx:
                    idx = val_to_idx[v]
                    for c in range(C):
                        lin[j, v, c] = w[col_idx + idx, c]
                        lat[j, v, :, c] = V[col_idx + idx, :, c]
            col_idx += len(vals)

    # Write weights.bin
    MAGIC = 0x464D3132
    header = struct.pack("<4i", MAGIC, d, best_k, C)
    body = w0.astype("<f4").tobytes() + lin.astype("<f4").tobytes() + lat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class index (0 or 1).
 * Weights format: magic 0x464D3132, d, k, C (int32);
 * float32 w0[C], lin[d][256][C], lat[d][256][k][C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *lin = w0 + C;
    const float *lat = lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_f = malloc(sizeof(float) * k);
    float *sum_sq_f = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t reclen; const unsigned char *xi = next_rec(x, nx, &pos, &reclen);
        if ((int)reclen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, reclen, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int f = 0; f < k; f++) { sum_f[f] = 0.0f; sum_sq_f[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_jv = lin + ((long)j * 256 + v) * C;
            const float *lat_jv = lat + ((long)j * 256 + v) * k * C;
            for (int c = 0; c < C; c++) scores[c] += lin_jv[c];
            for (int f = 0; f < k; f++) {
                for (int c = 0; c < C; c++) {
                    float val = lat_jv[(long)f * C + c];
                    sum_f[f] += val;
                    sum_sq_f[f] += val * val;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < k; f++) {
                scores[c] += 0.5f * (sum_f[f] * sum_f[f] - sum_sq_f[f]);
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```