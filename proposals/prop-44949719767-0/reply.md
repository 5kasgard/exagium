```hypothesis
Family: label-emb-linear (embedding + linear). Ordinal features are z-scored; each categorical feature gets a learned embedding vector (size E=4). A single linear layer on the concatenated embeddings and scaled ordinals predicts the class. This captures category similarities better than one-hot while adding only ~33 KB weights and ~38 multiply-adds per item. It should beat label-linear (0.8524) on dev accuracy; if dev accuracy ≤ 0.8524 the hypothesis fails.
```

```python
"""label-emb-linear: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, C, n_ord, n_cat
  float32 ord_mean[n_ord], ord_std[n_ord]
  float32 cat_emb[n_cat][256][E]
  float32 W[(n_ord + n_cat*E)][C], b[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
E_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03, 0.1]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
TUNE_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    # Split train/tune (last 20% in file order)
    n = X_bytes.shape[0]
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_te = X_bytes[:split], X_bytes[split:]
    y_tr, y_te = y[:split], y[split:]

    # Ordinal stats on full train (for final model) and on train split (for tuning)
    def compute_ord_stats(X):
        mu = X[:, ord_idx].mean(axis=0).astype(np.float32)
        sd = X[:, ord_idx].std(axis=0).astype(np.float32)
        sd[sd == 0] = 1.0
        return mu, sd

    mu_tr, sd_tr = compute_ord_stats(X_tr)
    mu_full, sd_full = compute_ord_stats(X_bytes)

    def scale_ord(X, mu, sd):
        return (X[:, ord_idx].astype(np.float32) - mu) / sd

    def get_cat_indices(X):
        return X[:, cat_idx].astype(np.int32)  # shape (n, n_cat)

    def train_eval(E, lr, X_tr, y_tr, X_te, y_te, mu, sd):
        rng = np.random.default_rng(SEED)
        cat_emb = rng.normal(0, 0.01, (n_cat, 256, E)).astype(np.float32)
        in_dim = n_ord + n_cat * E
        W = rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_b, v_b = np.zeros_like(b), np.zeros_like(b)
        m_emb = [np.zeros_like(cat_emb[i]) for i in range(n_cat)]
        v_emb = [np.zeros_like(cat_emb[i]) for i in range(n_cat)]
        t = 0
        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                # Forward
                z_ord = scale_ord(xb, mu, sd)  # (B, n_ord)
                cat_ids = get_cat_indices(xb)  # (B, n_cat)
                emb = np.concatenate([cat_emb[i][cat_ids[:, i]] for i in range(n_cat)], axis=1)  # (B, n_cat*E)
                z = np.concatenate([z_ord, emb], axis=1)  # (B, in_dim)
                logits = z @ W + b
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(len(yb)), yb] -= 1.0
                grad = probs / len(yb)  # (B, C)
                # Backward
                gW = z.T @ grad + L2 * W
                gb = grad.sum(axis=0)
                gz = grad @ W.T  # (B, in_dim)
                gz_ord = gz[:, :n_ord]
                gz_emb = gz[:, n_ord:].reshape(-1, n_cat, E)
                # Adam updates
                t += 1
                beta1, beta2 = 0.9, 0.999
                for i in range(n_cat):
                    gi = np.zeros_like(cat_emb[i])
                    np.add.at(gi, cat_ids[:, i], gz_emb[:, i])
                    m_emb[i] = beta1 * m_emb[i] + (1 - beta1) * gi
                    v_emb[i] = beta2 * v_emb[i] + (1 - beta2) * (gi * gi)
                    m_hat = m_emb[i] / (1 - beta1 ** t)
                    v_hat = v_emb[i] / (1 - beta2 ** t)
                    cat_emb[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * (gW * gW)
                m_b = beta1 * m_b + (1 - beta1) * gb
                v_b = beta2 * v_b + (1 - beta2) * (gb * gb)
                W -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + 1e-8)
                b -= lr * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + 1e-8)
        # Evaluate on tune
        z_ord_te = scale_ord(X_te, mu, sd)
        cat_ids_te = get_cat_indices(X_te)
        emb_te = np.concatenate([cat_emb[i][cat_ids_te[:, i]] for i in range(n_cat)], axis=1)
        z_te = np.concatenate([z_ord_te, emb_te], axis=1)
        logits_te = z_te @ W + b
        preds = logits_te.argmax(axis=1)
        acc = (preds == y_te).mean()
        return acc, (cat_emb, W, b)

    # Grid search on tuning split
    best_acc = -1
    best_params = None
    best_state = None
    for E in E_CHOICES:
        for lr in LR_CHOICES:
            acc, state = train_eval(E, lr, X_tr, y_tr, X_te, y_te, mu_tr, sd_tr)
            if acc > best_acc:
                best_acc = acc
                best_params = (E, lr)
                best_state = state
    E, lr = best_params
    cat_emb, W, b = best_state
    # Retrain on full data with best params
    _, (cat_emb, W, b) = train_eval(E, lr, X_bytes, y, X_bytes[:1], y[:1], mu_full, sd_full)

    # Write weights.bin
    magic = 0x454D424C
    header = struct.pack("<5i", magic, d, E, C, n_ord) + struct.pack("<i", n_cat)
    ord_stats = np.concatenate([mu_full, sd_full]).astype("<f4").tobytes()
    cat_emb_bytes = np.stack(cat_emb).astype("<f4").tobytes()  # (n_cat, 256, E)
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + ord_stats + cat_emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (14). Output record: 1 byte (class 0/1).
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d, E, C, n_ord, n_cat
 *   float32 ord_mean[n_ord], ord_std[n_ord]
 *   float32 cat_emb[n_cat][256][E]
 *   float32 W[(n_ord + n_cat*E)][C], b[C] */
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], n_ord = hd[4], n_cat = hd[5];
    if (C != 2 || d != 14 || n_ord != 6 || n_cat != 8 || E < 1 || E > 8) { fprintf(stderr, "bad header\n"); return 3; }
    int in_dim = n_ord + n_cat * E;
    long expected = 24L + 2L * n_ord * 4 + (long)n_cat * 256 * E * 4 + (long)in_dim * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ord_mean = (float *)(hd + 6);
    float *ord_std = ord_mean + n_ord;
    float *cat_emb = ord_std + n_ord;  // [n_cat][256][E]
    float *W = cat_emb + (long)n_cat * 256 * E;
    float *b = W + (long)in_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * in_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Ordinal features (indices 0,2,4,10,11,12)
        int ord_map[6] = {0,2,4,10,11,12};
        for (int j = 0; j < n_ord; j++) {
            z[j] = (xi[ord_map[j]] - ord_mean[j]) / ord_std[j];
        }
        // Categorical features (indices 1,3,5,6,7,8,9,13)
        int cat_map[8] = {1,3,5,6,7,8,9,13};
        for (int c = 0; c < n_cat; c++) {
            int idx = xi[cat_map[c]];
            const float *emb = cat_emb + ((long)c * 256 + idx) * E;
            for (int e = 0; e < E; e++) {
                z[n_ord + c * E + e] = emb[e];
            }
        }
        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < in_dim; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * W[(long)j * C + c];
        }
        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```