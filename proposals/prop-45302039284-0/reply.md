```hypothesis
Family: label-emb (embedding + linear).  Ordinal columns are z-scored; each categorical column gets a learned embedding vector (size E) plus an extra row for unseen (255).  The concatenated embeddings and ordinal scalars feed a single logistic-regression layer.  Tuning chooses E ∈ {2,4,8,16} on the last 20 % of train.in, then retrains on all data.  This captures categorical interactions cheaply and should beat label-linear (0.8524) with similar inference cost (only a few extra additions per categorical feature).  If dev accuracy ≤ 0.853 or Ir/item > 1000 the idea fails.
```
```python
"""label-emb: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [2, 4, 8, 16]          # tuned on last 20% of train.in
MAGIC = 0x454D4245                      # "EMBE" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_vocabs(X, types):
    """Return list of (is_cat, vocab_size) for each column."""
    info = []
    for j, t in enumerate(types):
        if t == "categorical":
            vals = np.unique(X[:, j])
            info.append((True, len(vals)))
        else:
            info.append((False, 0))
    return info


def make_embedding_tables(vocabs, E, rng):
    """List of float32 arrays: (vocab_size+1, E) for cat, None for ord."""
    tables = []
    for is_cat, vocab_size in vocabs:
        if is_cat:
            W = (rng.uniform(-1, 1, (vocab_size + 1, E)) *
                 math.sqrt(6.0 / (vocab_size + E))).astype(np.float32)
            tables.append(W)
        else:
            tables.append(None)
    return tables


def encode(X, types, vocabs, tables, ord_stats):
    """Convert raw byte matrix X to dense float32 matrix Z."""
    cols = []
    for j, t in enumerate(types):
        is_cat, vocab_size = vocabs[j]
        if is_cat:
            idx = X[:, j].astype(np.int32)
            idx[idx > vocab_size] = vocab_size          # unseen -> last row
            cols.append(tables[j][idx])
        else:
            mu, sd = ord_stats[j]
            cols.append(((X[:, j].astype(np.float32) - mu) / sd)[:, None])
    return np.concatenate(cols, axis=1)


def train_logreg(Z, y, C, rng, epochs):
    """Adam on linear softmax. Returns W (k,C), b (C)."""
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(0)
            t += 1
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            lr_t = LR / (1 - 0.9 ** t)
            W -= lr_t * mW / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
            b -= lr_t * mb / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def accuracy(Z, y, W, b):
    logits = Z @ W + b
    return (logits.argmax(1) == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    vocabs = build_vocabs(X, types)
    ord_stats = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_stats.append((float(mu), float(sd)))
        else:
            ord_stats.append((0.0, 1.0))

    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)

    # ----- tuning -----
    best_E, best_acc = E_CANDIDATES[0], -1.0
    for E in E_CANDIDATES:
        tables = make_embedding_tables(vocabs, E, np.random.default_rng(SEED + E))
        Z_tr = encode(X[:split], types, vocabs, tables, ord_stats)
        Z_va = encode(X[split:], types, vocabs, tables, ord_stats)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
        W, b = train_logreg(Z_tr, y[:split], C, np.random.default_rng(SEED + E + 100), epochs)
        acc = accuracy(Z_va, y[split:], W, b)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # ----- final training on all data -----
    tables = make_embedding_tables(vocabs, best_E, np.random.default_rng(SEED + best_E))
    Z_all = encode(X, types, vocabs, tables, ord_stats)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, b = train_logreg(Z_all, y, C, np.random.default_rng(SEED + best_E + 100), epochs)

    # ----- serialize weights -----
    # Layout:
    #   int32 magic, d, E, C, num_cat
    #   for each feature: int32 is_cat, int32 vocab_size
    #   for each feature:
    #       if cat: float32[(vocab_size+1)*E] embedding table (row-major)
    #       if ord: float32[2] mean, std
    #   float32[total_dim*C] W (row-major)
    #   float32[C] b
    num_cat = sum(1 for is_cat, _ in vocabs if is_cat)
    total_dim = num_cat * best_E + (d - num_cat)

    out = bytearray()
    out += struct.pack("<5i", MAGIC, d, best_E, C, num_cat)
    for is_cat, vocab_size in vocabs:
        out += struct.pack("<2i", 1 if is_cat else 0, vocab_size)
    for j, (is_cat, vocab_size) in enumerate(vocabs):
        if is_cat:
            out += tables[j].astype("<f4").tobytes()
        else:
            mu, sd = ord_stats[j]
            out += struct.pack("<2f", mu, sd)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record = d bytes (d from header). Output = 1 byte class index. */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 20 || hdr[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], E = hdr[2], C = hdr[3], num_cat = hdr[4];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad header values\n"); return 3; }

    int32_t *feat = hdr + 5;                     // 2 ints per feature
    if (nw < 20 + 8L * d) { fprintf(stderr, "weights too small for feature table\n"); return 3; }

    /* compute offsets */
    int *is_cat = malloc(d * sizeof(int));
    int *vocab = malloc(d * sizeof(int));
    int total_dim = 0;
    for (int j = 0; j < d; j++) {
        is_cat[j] = feat[2*j];
        vocab[j] = feat[2*j+1];
        if (is_cat[j]) {
            if (vocab[j] < 0) { fprintf(stderr, "bad vocab size\n"); return 3; }
            total_dim += E;
        } else {
            total_dim += 1;
        }
    }

    /* embedding tables and ord stats */
    float *emb_base = (float *)(feat + 2 * d);
    float *emb_ptr = emb_base;
    float **emb = malloc(d * sizeof(float *));
    float *ord_mu = malloc(d * sizeof(float));
    float *ord_sd = malloc(d * sizeof(float));
    for (int j = 0; j < d; j++) {
        if (is_cat[j]) {
            emb[j] = emb_ptr;
            emb_ptr += (long)(vocab[j] + 1) * E;
        } else {
            emb[j] = NULL;
            ord_mu[j] = emb_ptr[0];
            ord_sd[j] = emb_ptr[1];
            emb_ptr += 2;
        }
    }
    float *W = emb_ptr;
    float *b = W + (long)total_dim * C;

    /* validate total size */
    long expected = (char *)b + C * sizeof(float) - (char *)wb;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(total_dim * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int w = 0;
        for (int j = 0; j < d; j++) {
            if (is_cat[j]) {
                int idx = xi[j];
                if (idx > vocab[j]) idx = vocab[j];      // unseen
                const float *row = emb[j] + (long)idx * E;
                for (int e = 0; e < E; e++) vec[w++] = row[e];
            } else {
                vec[w++] = (xi[j] - ord_mu[j]) / ord_sd[j];
            }
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int k = 0; k < total_dim; k++) sum += vec[k] * W[(long)k * C + c];
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```