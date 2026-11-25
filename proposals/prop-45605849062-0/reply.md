```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from pairwise feature interactions (e.g., education×occupation, marital-status×sex) which a linear model misses. A low-rank FM (k=8) captures these with O(k·D) parameters and compute, staying faster than an MLP while improving accuracy. If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear without accuracy gain, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-score, categorical as one-hot (unseen=0).
Tunes k, lr, l2 on last 20% of train.in, then trains on all data.
Weights format: magic 0x464D4C31, d, D, k, C; int32 D_j[d]; float32 E_j[256][D_j]; float32 w0, w[D], V[D][k]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
K_GRID = [4, 8]
LR_GRID = [0.01, 0.03]
L2_GRID = [1e-4, 1e-3]
BATCH = 64
EPOCHS_TUNE = 50
EPOCHS_FULL = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x D_j) and list D_j."""
    enc, D_j = [], []
    for j, t in enumerate(types):
        col = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
            D_j.append(1)
        else:
            vals = np.unique(X[:, j])
            V = len(vals)
            E = np.zeros((256, V), dtype=np.float32)
            E[vals, np.arange(V)] = 1.0
            enc.append(E)
            D_j.append(V)
    return enc, D_j


def encode_all(X, enc):
    """Concatenate encoded columns into feature matrix Z (n x D)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_fm(Z, y, D, k, lr, l2, seed, epochs):
    """Train FM with Adam, return (w0, w, V)."""
    rng = np.random.default_rng(seed)
    w0 = np.float32(0.0)
    w = np.zeros(D, dtype=np.float32)
    V = rng.normal(0, 0.01, (D, k)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = Z.shape[0]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = Z[idx]
            yy = y[idx]
            b = x.shape[0]

            linear = w0 + x @ w
            XV = x @ V
            x2 = x * x
            V2 = V * V
            interaction = 0.5 * np.sum(XV * XV - x2 @ V2, axis=1)
            logits = linear + interaction

            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = p - yy

            g_w0 = grad_logits.mean()
            g_w = (x.T @ grad_logits) / b + l2 * w
            g_V = (x.T @ (grad_logits[:, None] * XV)) / b - (x2.T @ (grad_logits[:, None] * V)) / b + l2 * V

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    return w0, w, V


def predict_fm(Z, model):
    w0, w, V = model
    linear = w0 + Z @ w
    XV = Z @ V
    x2 = Z * Z
    V2 = V * V
    interaction = 0.5 * np.sum(XV * XV - x2 @ V2, axis=1)
    return (linear + interaction > 0).astype(np.int64)


def save_weights(path, d, D, k, C, D_j, enc, model):
    w0, w, V = model
    magic = 0x464D4C31  # 'FML1'
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", magic, d, D, k, C))
        f.write(struct.pack(f"<{d}i", *D_j))
        for E in enc:
            f.write(E.astype("<f4").tobytes())
        f.write(struct.pack("<f", w0))
        f.write(w.astype("<f4").tobytes())
        f.write(V.astype("<f4").tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]

    enc, D_j = build_encoding(X_bytes, types)
    D = sum(D_j)
    Z = encode_all(X_bytes, enc)

    n = Z.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    Z_tr, Z_te = Z[:train_size], Z[train_size:]
    y_tr, y_te = y[:train_size], y[train_size:]

    best_acc, best_params, best_model = -1, None, None
    for k in K_GRID:
        for lr in LR_GRID:
            for l2 in L2_GRID:
                model = train_fm(Z_tr, y_tr, D, k, lr, l2, SEED, EPOCHS_TUNE)
                preds = predict_fm(Z_te, model)
                acc = (preds == y_te).mean()
                if acc > best_acc:
                    best_acc, best_params, best_model = acc, (k, lr, l2), model

    k, lr, l2 = best_params
    final_model = train_fm(Z, y, D, k, lr, l2, SEED, EPOCHS_FULL)
    save_weights(sys.argv[4], d, D, k, C, D_j, enc, final_model)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], D = hdr[2], k = hdr[3], C = hdr[4];
    if (d <= 0 || D <= 0 || k <= 0 || C != 2) { fprintf(stderr, "bad dims\n"); return 3; }

    int32_t *Dj = (int32_t *)(hdr + 5);
    long expected = 20 + 4L * d;                       // header + Dj
    for (int j = 0; j < d; j++) {
        if (Dj[j] <= 0 || Dj[j] > 256) { fprintf(stderr, "bad Dj[%d]\n", j); return 3; }
        expected += 4L * 256 * Dj[j];                  // encoding matrices
    }
    expected += 4L * (1 + D + (long)D * k);            // w0, w, V
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    float *enc = (float *)(Dj + d);
    float *w0 = (float *)((unsigned char *)enc + 4L * 256 * D);
    float *w = w0 + 1;
    float *V = w + D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *XV = malloc(sizeof(float) * k);
    float *V2 = malloc(sizeof(float) * D * k);
    for (long i = 0; i < (long)D * k; i++) V2[i] = V[i] * V[i];

    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *rec = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }

        int off = 0;
        for (int j = 0; j < d; j++) {
            const float *row = enc + (long)rec[j] * Dj[j];
            memcpy(feat + off, row, sizeof(float) * Dj[j]);
            off += Dj[j];
        }

        float linear = w0[0];
        for (int j = 0; j < D; j++) linear += feat[j] * w[j];

        for (int f = 0; f < k; f++) {
            float sum = 0.0f;
            for (int j = 0; j < D; j++) sum += feat[j] * V[(long)j * k + f];
            XV[f] = sum;
        }

        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            float sq = XV[f] * XV[f];
            float sum2 = 0.0f;
            for (int j = 0; j < D; j++) sum2 += feat[j] * feat[j] * V2[(long)j * k + f];
            interaction += 0.5f * (sq - sum2);
        }

        unsigned char pred = (linear + interaction > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(feat); free(XV); free(V2);
    if (x) free(x); free(wb);
    return 0;
}
```