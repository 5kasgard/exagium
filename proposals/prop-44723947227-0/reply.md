```hypothesis
Family: label-embed (embedding + linear). Ordinal features are z-scored; each categorical feature gets a learned embedding vector of dimension E. A single linear layer on the concatenated embeddings and ordinal features predicts the class. This captures categorical similarities with far fewer parameters than one-hot, should match or beat label-linear accuracy (0.8524) at similar or lower inference cost because the embedding lookup is just a memory read and the linear layer is small (D ≈ 6 + 8*E). If dev accuracy stays below 0.85 or inference cost exceeds label-linear by >20%, the hypothesis fails.
```
```python
"""label-embed: ordinal z-scored + categorical embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Tuning grid (chosen on last 20% of train.in)
EMBED_DIMS = [4, 8, 16]
LRS = [1e-3, 3e-3]
EPOCHS_GRID = [30, 50]
BATCH = 64
SEED = 1
VAL_FRAC = 0.2  # last 20% in file order

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, ordinal_idx, cat_idx, mean, std, E, lr, epochs, rng, C):
    n, d = X.shape
    O = len(ordinal_idx)
    K = len(cat_idx)
    D = O + K * E
    # Initialize embeddings for each categorical feature: (256, E)
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(K)]
    W = rng.normal(0, 0.01, (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = emb + [W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(Xb):
        # Xb: (bs, d) uint8
        bs = Xb.shape[0]
        Z = np.empty((bs, D), np.float32)
        # ordinal
        if O:
            Z[:, :O] = (Xb[:, ordinal_idx].astype(np.float32) - mean) / std
        # categorical embeddings
        off = O
        for ki, cj in enumerate(cat_idx):
            Z[:, off:off+E] = emb[ki][Xb[:, cj]]
            off += E
        scores = Z @ W + b
        # softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return Z, probs

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            Z, probs = forward(Xb)
            # gradient of cross-entropy
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # grads
            gW = Z.T @ probs
            gb = probs.sum(axis=0)
            gZ = probs @ W.T
            # embedding grads
            gemb = [np.zeros_like(e) for e in emb]
            off = O
            for ki, cj in enumerate(cat_idx):
                np.add.at(gemb[ki], Xb[:, cj], gZ[:, off:off+E])
                off += E
            grads = gemb + [gW, gb]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return emb, W, b

def evaluate(X, y, ordinal_idx, cat_idx, mean, std, emb, W, b, C):
    n = X.shape[0]
    O = len(ordinal_idx)
    K = len(cat_idx)
    E = emb[0].shape[1]
    D = O + K * E
    Z = np.empty((n, D), np.float32)
    if O:
        Z[:, :O] = (X[:, ordinal_idx].astype(np.float32) - mean) / std
    off = O
    for ki, cj in enumerate(cat_idx):
        Z[:, off:off+E] = emb[ki][X[:, cj]]
        off += E
    scores = Z @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    ordinal_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    # compute ordinal stats on full training set (will be used for final model)
    mean_full = X[:, ordinal_idx].mean(axis=0).astype(np.float32)
    std_full = X[:, ordinal_idx].std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0

    # split for tuning: last 20% in file order
    n = X.shape[0]
    val_size = int(n * VAL_FRAC)
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    # ordinal stats on training split only
    mean_tr = X_train[:, ordinal_idx].mean(axis=0).astype(np.float32)
    std_tr = X_train[:, ordinal_idx].std(axis=0).astype(np.float32)
    std_tr[std_tr == 0] = 1.0

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_emb = best_W = best_b = None

    for E in EMBED_DIMS:
        for lr in LRS:
            for ep in EPOCHS_GRID:
                # train on training split for fewer epochs for tuning? Use same epochs.
                emb, W, b = train_model(X_train, y_train, ordinal_idx, cat_idx, mean_tr, std_tr, E, lr, ep, rng, C)
                acc = evaluate(X_val, y_val, ordinal_idx, cat_idx, mean_tr, std_tr, emb, W, b, C)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (E, lr, ep)
                    best_emb, best_W, best_b = emb, W, b
                print(f"Tune E={E} lr={lr} ep={ep} val_acc={acc:.4f}", file=sys.stderr)

    print(f"Best config: E={best_cfg[0]} lr={best_cfg[1]} ep={best_cfg[2]} val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    rng_full = np.random.default_rng(SEED)
    final_emb, final_W, final_b = train_model(X, y, ordinal_idx, cat_idx, mean_full, std_full,
                                              best_cfg[0], best_cfg[1], best_cfg[2], rng_full, C)

    # Write weights.bin
    d = X.shape[1]
    E = best_cfg[0]
    magic = 0x454D4244  # "EMBD"
    type_bytes = bytes([0 if t == "ordinal" else 1 for t in types])
    # compute D
    O = len(ordinal_idx)
    K = len(cat_idx)
    D = O + K * E
    # pack
    out = struct.pack("<4I", magic, d, E, C)
    out += type_bytes
    for t in types:
        if t == "ordinal":
            idx = ordinal_idx.pop(0)  # but we need original order
    # We need to write in feature order 0..d-1
    ordinal_iter = iter(ordinal_idx)
    cat_iter = iter(cat_idx)
    for t in types:
        if t == "ordinal":
            j = next(ordinal_iter)
            out += struct.pack("<ff", mean_full[j], std_full[j])
        else:
            j = next(cat_iter)
            out += final_emb[cat_idx.index(j)].astype("<f4").tobytes()
    out += final_W.astype("<f4").tobytes()
    out += final_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 per feature). Output: one byte class index. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    unsigned char *types = wb + 16;
    if (nw < 16 + d) { fprintf(stderr, "truncated types\n"); return 3; }
    const float *ptr = (const float *)(types + d);
    float *mean = malloc(d * sizeof(float));
    float *std = malloc(d * sizeof(float));
    const float **emb = malloc(d * sizeof(float*));
    int O = 0, K = 0;
    for (int j = 0; j < d; j++) {
        if (types[j] == 0) { /* ordinal */
            if ((char*)ptr + 8 > wb + nw) { fprintf(stderr, "truncated ordinal params\n"); return 3; }
            mean[j] = ptr[0]; std[j] = ptr[1];
            ptr += 2;
            O++;
        } else { /* categorical */
            if ((char*)ptr + 256 * E * 4 > wb + nw) { fprintf(stderr, "truncated embedding\n"); return 3; }
            emb[j] = ptr;
            ptr += 256 * E;
            K++;
        }
    }
    int D = O + K * E;
    if ((char*)ptr + (long)D * C * 4 + C * 4 > wb + nw) { fprintf(stderr, "truncated linear\n"); return 3; }
    const float *W = ptr; ptr += (long)D * C;
    const float *b = ptr;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *Z = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int off = 0;
        for (int j = 0; j < d; j++) {
            if (types[j] == 0) {
                Z[off++] = (xi[j] - mean[j]) / std[j];
            } else {
                const float *e = emb[j] + xi[j] * E;
                for (int u = 0; u < E; u++) Z[off++] = e[u];
            }
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < D; u++) {
            float zu = Z[u];
            if (zu != 0.0f) {
                for (int c = 0; c < C; c++) scores[c] += zu * W[(long)u * C + c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```