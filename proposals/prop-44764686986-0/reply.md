```hypothesis
Family: label-emblin (embedding + linear). Ordinal features are z-scored; each categorical feature gets a learned embedding vector of dimension E (tuned on last 20% of train). The concatenated embeddings and ordinal values feed a linear output layer. This should capture categorical similarities better than one-hot linear, and with small E (e.g., 4) it reduces input dimension (38 vs ~3000) for faster inference. If dev accuracy ≤ label-linear (0.8524) or Ir/item ≥ label-linear (719), the hypothesis fails.
```
```python
"""label-emblin: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x4C424D45 ('EMBL')
  int32 d, int32 E, int32 C, int32 n_ord, int32 n_cat
  char types[d] (0=ordinal,1=categorical)
  float32 ord_mean[n_ord], ord_std[n_ord]
  float32 cat_emb[n_cat * 256 * E]
  float32 W_out[(n_ord + n_cat*E) * C]
  float32 b_out[C]"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [2, 4, 8, 16]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)
    n_ord = sum(1 for t in types if t == "ordinal")
    n_cat = d - n_ord
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]

    # Split train (first 80%) / tune (last 20%) in file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    # Precompute ordinal stats on full training set (as per card: choose settings on tuning, then train on all)
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Categorical cardinalities (including unseen=255)
    cat_cards = [int(X[:, j].max()) + 1 for j in cat_idx]
    cat_cards = [max(c, 256) for c in cat_cards]  # ensure 256 for unseen

    best_E, best_acc = 4, -1.0
    rng = np.random.default_rng(SEED)

    for E in E_CANDIDATES:
        # Initialize embeddings and output layer
        cat_emb = [rng.uniform(-0.1, 0.1, (card, E)).astype(np.float32) for card in cat_cards]
        total_dim = n_ord + n_cat * E
        W_out = rng.uniform(-0.1, 0.1, (total_dim, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = cat_emb + [W_out, b_out]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

        # Training on train split
        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                bs = len(yb)

                # Forward
                h = np.zeros((bs, total_dim), np.float32)
                # Ordinal part
                if n_ord:
                    h[:, :n_ord] = ((xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)
                # Categorical embeddings
                off = n_ord
                for ci, j in enumerate(cat_idx):
                    vals = xb[:, j]
                    h[:, off:off + E] = cat_emb[ci][vals]
                    off += E
                logits = h @ W_out + b_out
                # Softmax + cross-entropy gradient
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits, dtype=np.float32)
                probs = exp / exp.sum(axis=1, keepdims=True)
                probs[np.arange(bs), yb] -= 1.0
                probs /= bs

                # Gradients
                g_W = h.T @ probs + L2 * W_out
                g_b = probs.sum(axis=0)
                g_h = probs @ W_out.T

                # Embedding gradients (accumulate)
                g_emb = [np.zeros_like(e) for e in cat_emb]
                off = n_ord
                for ci, j in enumerate(cat_idx):
                    vals = xb[:, j]
                    np.add.at(g_emb[ci], vals, g_h[:, off:off + E])
                    g_emb[ci] += L2 * cat_emb[ci]
                    off += E
                if n_ord:
                    g_ord = g_h[:, :n_ord]  # not used directly, but ordinal params are fixed
                grads = g_emb + [g_W, g_b]

                # Adam update
                t = 1  # simplified: single global step counter per batch
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on tuning split
        bs = X_va.shape[0]
        h = np.zeros((bs, total_dim), np.float32)
        if n_ord:
            h[:, :n_ord] = ((X_va[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)
        off = n_ord
        for ci, j in enumerate(cat_idx):
            vals = X_va[:, j]
            h[:, off:off + E] = cat_emb[ci][vals]
            off += E
        logits = h @ W_out + b_out
        preds = logits.argmax(axis=1)
        acc = (preds == y_va).mean()
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on full training data with best_E
    E = best_E
    cat_emb = [rng.uniform(-0.1, 0.1, (card, E)).astype(np.float32) for card in cat_cards]
    total_dim = n_ord + n_cat * E
    W_out = rng.uniform(-0.1, 0.1, (total_dim, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = cat_emb + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n_tr = n
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            h = np.zeros((bs, total_dim), np.float32)
            if n_ord:
                h[:, :n_ord] = ((xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)
            off = n_ord
            for ci, j in enumerate(cat_idx):
                vals = xb[:, j]
                h[:, off:off + E] = cat_emb[ci][vals]
                off += E
            logits = h @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits, dtype=np.float32)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            g_W = h.T @ probs + L2 * W_out
            g_b = probs.sum(axis=0)
            g_h = probs @ W_out.T

            g_emb = [np.zeros_like(e) for e in cat_emb]
            off = n_ord
            for ci, j in enumerate(cat_idx):
                vals = xb[:, j]
                np.add.at(g_emb[ci], vals, g_h[:, off:off + E])
                g_emb[ci] += L2 * cat_emb[ci]
                off += E
            grads = g_emb + [g_W, g_b]

            t = 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    types_arr = bytearray(1 if t == "categorical" else 0 for t in types)
    magic = 0x4C424D45  # 'EMBL'
    header = struct.pack("<iiiii", magic, d, E, C, n_ord)
    header += struct.pack("<i", n_cat)
    header += types_arr
    header += ord_mean.astype("<f4").tobytes()
    header += ord_std.astype("<f4").tobytes()
    for e in cat_emb:
        header += e.astype("<f4").tobytes()
    header += W_out.astype("<f4").tobytes()
    header += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header)


if __name__ == "__main__":
    main()
```
```c
/* label-emblin inference: ./infer weights.bin in.rec N
 * Input record: d bytes (14). Output: 1 byte class index. */
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
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C424D45) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], n_ord = hd[4], n_cat = hd[5];
    if (d != 14 || C != 2 || E < 1 || E > 16 || n_ord + n_cat != d) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    unsigned char *types = wb + 24;
    float *ord_mean = (float *)(types + d);
    float *ord_std = ord_mean + n_ord;
    float *cat_emb = ord_std + n_ord;
    int total_dim = n_ord + n_cat * E;
    float *W_out = cat_emb;
    for (int i = 0; i < n_cat; i++) W_out += 256 * E;
    float *b_out = (float *)W_out + total_dim * C;
    /* Validate total size */
    long expected = 24 + d + (long)n_ord * 8 + (long)n_cat * 256 * E * 4 + (long)total_dim * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * total_dim);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* Build hidden vector */
        int hpos = 0;
        for (int j = 0; j < d; j++) {
            if (types[j] == 0) { /* ordinal */
                float v = (xi[j] - ord_mean[hpos]) / ord_std[hpos];
                h[hpos++] = v;
            } else { /* categorical */
                int cat = xi[j];
                const float *emb = cat_emb + (long)(j - n_ord) * 256 * E + cat * E;
                for (int e = 0; e < E; e++) h[hpos++] = emb[e];
            }
        }
        /* Output layer */
        for (int c = 0; c < C; c++) sc[c] = b_out[c];
        for (int j = 0; j < total_dim; j++) {
            float hj = h[j];
            for (int c = 0; c < C; c++) sc[c] += hj * W_out[(long)j * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```