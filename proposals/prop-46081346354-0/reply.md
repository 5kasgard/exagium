```hypothesis
Family: label-emb (embedding + linear). Categorical columns get learned dense embeddings (dim=4) instead of one-hot, ordinal columns stay z-scored. A single linear layer on top yields a compact model that captures category similarities. Expect dev accuracy > 0.8524 (label-linear) with similar inference cost (same lookup+dot structure). Failure if accuracy ≤ label-linear or weights file size grows > 2×.
```
```python
"""label-emb: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Categorical cols -> learned embeddings (dim E), ordinal cols -> z-scored.
Linear layer on concatenated embeddings. E chosen on last 20% of train.in.
Exported as per-byte lookup tables folded into first layer (same format as label-linear)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [2, 4, 8, 16]  # embedding dimensions to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, emb_tables, ord_params, W1, b1, cat_cols, ord_cols, E, C):
    """Compute accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        idx = slice(s, min(s + 1024, n))
        xb, yb = X[idx], y[idx]
        batch = xb.shape[0]
        feats = np.zeros((batch, len(cat_cols) * E + len(ord_cols)), dtype=np.float32)
        off = 0
        for j in cat_cols:
            feats[:, off:off + E] = emb_tables[j][xb[:, j]]
            off += E
        for j in ord_cols:
            mu, sd = ord_params[j]
            feats[:, off] = (xb[:, j].astype(np.float32) - mu) / sd
            off += 1
        logits = feats @ W1 + b1
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_once(X, y, cat_cols, ord_cols, E, C, rng, epochs, batch_size):
    """Train embeddings and linear layer, return (emb_tables, ord_params, W1, b1)."""
    n, d = X.shape
    # Build ordinal normalization params (fixed)
    ord_params = {}
    for j in ord_cols:
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        ord_params[j] = (mu, sd)
    # Initialize embeddings for categorical columns
    emb_tables = {}
    for j in cat_cols:
        emb_tables[j] = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # Linear layer
    D = len(cat_cols) * E + len(ord_cols)
    W1 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1] + [emb_tables[j] for j in cat_cols]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb, yb = X[idx], y[idx]
            batch = xb.shape[0]
            # Build feature matrix and keep track of which embedding rows used
            feats = np.zeros((batch, D), dtype=np.float32)
            off = 0
            cat_slices = []
            for j in cat_cols:
                feats[:, off:off + E] = emb_tables[j][xb[:, j]]
                cat_slices.append((j, off, off + E))
                off += E
            for j in ord_cols:
                mu, sd = ord_params[j]
                feats[:, off] = (xb[:, j].astype(np.float32) - mu) / sd
                off += 1
            # Forward
            logits = feats @ W1 + b1
            g = sg(logits, yb)  # (batch, C)
            # Gradients
            grad_W1 = feats.T @ g + L2 * W1
            grad_b1 = g.sum(0)
            grad_emb = {j: np.zeros((256, E), np.float32) for j in cat_cols}
            grad_feats = g @ W1.T  # (batch, D)
            off = 0
            for j in cat_cols:
                grad_emb[j][xb[:, j]] += grad_feats[:, off:off + E]
                off += E
            off += len(ord_cols)  # skip ordinal (no learnable params)
            # Adam update
            grads = [grad_W1, grad_b1] + [grad_emb[j] for j in cat_cols]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return emb_tables, ord_params, W1, b1


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]
    n = X.shape[0]
    split = int(0.8 * n)  # last 20% for tuning
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))
    best_E, best_acc = E_CANDIDATES[0], -1.0
    for E in E_CANDIDATES:
        emb_tables, ord_params, W1, b1 = train_once(X_tr, y_tr, cat_cols, ord_cols, E, C, rng, epochs, BATCH)
        acc = evaluate(X_val, y_val, emb_tables, ord_params, W1, b1, cat_cols, ord_cols, E, C)
        if acc > best_acc:
            best_acc, best_E = acc, E
    # Retrain on full data with best_E
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb_tables, ord_params, W1, b1 = train_once(X, y, cat_cols, ord_cols, best_E, C, rng, epochs, BATCH)
    # Export: fold first layer into per-byte lookup tables T[j][v][c] = embedding[j][v] @ W1_slice + (ordinal part)
    T = np.zeros((X.shape[1], 256, C), dtype=np.float32)
    off = 0
    for j in range(X.shape[1]):
        if j in cat_cols:
            sl = slice(off, off + best_E)
            T[j] = emb_tables[j] @ W1[sl]
            off += best_E
        else:
            mu, sd = ord_params[j]
            vals = ((np.arange(256) - mu) / sd).astype(np.float32)[:, None]
            T[j] = vals @ W1[off:off + 1]
            off += 1
    # Pack weights: magic, d, H=0, C; then T (d*256*C floats), then b1 (C floats)
    magic = 0x454D424C  # 'EMBL'
    out = struct.pack("<4i", magic, X.shape[1], 0, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb infer: ./infer weights.bin in.rec N
 * Weights format: magic 0x454D424C, int32 d, H=0, C; float32 T[d][256][C]; float32 b1[C].
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) sc[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```