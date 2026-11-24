```hypothesis
Family: label-tenc-linear (target-encoded linear model). Categorical features are replaced by smoothed target means (one float per category), ordinal features are z-scored. A single linear layer follows. This reduces dimensionality vs one-hot, captures category-level signal non-linearly, and keeps inference at one dot product per item. If dev accuracy does not exceed label-linear by >0.3% (i.e., ≤0.855) we reject the family.
```

```python
"""label-tenc-linear: target encoding + linear model.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
ALPHA_GRID = [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]  # smoothing strengths


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
    n, d = X_bytes.shape
    types = card["input"]["types"]
    C = card["classes"]

    # split: last 20% for tuning
    split = int(n * 0.8)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X_bytes[idx_tr], y[idx_tr]
    X_va, y_va = X_bytes[idx_va], y[idx_va]

    # identify categorical and ordinal columns
    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]

    # global target mean for smoothing
    global_mean = y_tr.mean()

    # choose best alpha on validation
    best_alpha, best_acc = ALPHA_GRID[0], -1.0
    for alpha in ALPHA_GRID:
        # compute target encodings on training part
        enc_tables = []
        for j in cat_cols:
            table = np.full(256, global_mean, dtype=np.float32)
            for v in range(256):
                mask = (X_tr[:, j] == v)
                cnt = mask.sum()
                if cnt:
                    table[v] = (y_tr[mask].sum() + alpha * global_mean) / (cnt + alpha)
            enc_tables.append(table)
        # ordinal stats
        ord_means = X_tr[:, ord_cols].mean(axis=0).astype(np.float32)
        ord_stds = X_tr[:, ord_cols].std(axis=0).astype(np.float32)
        ord_stds[ord_stds == 0] = 1.0

        # build design matrices
        def encode(X):
            parts = []
            for idx, j in enumerate(cat_cols):
                parts.append(enc_tables[idx][X[:, j]].astype(np.float32)[:, None])
            if ord_cols:
                parts.append(((X[:, ord_cols].astype(np.float32) - ord_means) / ord_stds))
            return np.concatenate(parts, axis=1)

        Z_tr = encode(X_tr)
        Z_va = encode(X_va)
        k = Z_tr.shape[1]

        # train linear model on training part
        rng = np.random.default_rng(SEED)
        W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
        b = np.zeros(C, np.float32)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        t = 0
        for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_tr)))):
            order = rng.permutation(len(Z_tr))
            for s in range(0, len(Z_tr), BATCH):
                idx = order[s:s + BATCH]
                z, yy = Z_tr[idx], y_tr[idx]
                S = z @ W + b
                S -= S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                grad = (P / len(yy)).astype(np.float32)
                gW = z.T @ grad + L2 * W
                gb = grad.sum(0)
                t += 1
                for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                    m *= 0.9; m += 0.1 * grad
                    v *= 0.999; v += 0.001 * grad * grad
                    param -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

        # evaluate on validation
        scores = Z_va @ W + b
        preds = scores.argmax(axis=1)
        acc = (preds == y_va).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # retrain on full data with best_alpha
    enc_tables = []
    for j in cat_cols:
        table = np.full(256, global_mean, dtype=np.float32)
        for v in range(256):
            mask = (X_bytes[:, j] == v)
            cnt = mask.sum()
            if cnt:
                table[v] = (y[mask].sum() + best_alpha * global_mean) / (cnt + best_alpha)
        enc_tables.append(table)
    ord_means = X_bytes[:, ord_cols].mean(axis=0).astype(np.float32)
    ord_stds = X_bytes[:, ord_cols].std(axis=0).astype(np.float32)
    ord_stds[ord_stds == 0] = 1.0

    def encode_full(X):
        parts = []
        for idx, j in enumerate(cat_cols):
            parts.append(enc_tables[idx][X[:, j]].astype(np.float32)[:, None])
        if ord_cols:
            parts.append(((X[:, ord_cols].astype(np.float32) - ord_means) / ord_stds))
        return np.concatenate(parts, axis=1)

    Z = encode_full(X_bytes)
    k = Z.shape[1]
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            grad = (P / len(yy)).astype(np.float32)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(0)
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                param -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights.bin
    # magic 0x54454E43 ("TENC"), d, C, n_cat, n_ord
    # then for each cat col: 256 float32
    # then for each ord col: mean float32, std float32
    # then W[k, C] float32, b[C] float32
    magic = 0x54454E43
    n_cat, n_ord = len(cat_cols), len(ord_cols)
    header = struct.pack("<5i", magic, d, C, n_cat, n_ord)
    cat_bytes = b"".join(t.astype("<f4").tobytes() for t in enc_tables)
    ord_bytes = b"".join(struct.pack("<2f", m, s) for m, s in zip(ord_means, ord_stds))
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + cat_bytes + ord_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-tenc-linear infer.c
 * ./infer weights.bin in.rec N
 * weights.bin format: magic(0x54454E43), d, C, n_cat, n_ord,
 *   cat_tables[n_cat][256] float32,
 *   ord_stats[n_ord][2] float32 (mean, std),
 *   W[k][C] float32, b[C] float32
 * where k = n_cat + n_ord.
 * Input record: d bytes. Output: one byte class index. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x54454E43) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], n_cat = hdr[3], n_ord = hdr[4];
    if (C < 1 || C > 256 || n_cat < 0 || n_ord < 0 || n_cat + n_ord == 0) { fprintf(stderr, "bad dims\n"); return 3; }
    int k = n_cat + n_ord;
    const float *ptr = (const float *)(hdr + 5);
    const float *cat_tables = ptr; ptr += (long)n_cat * 256;
    const float *ord_stats = ptr; ptr += (long)n_ord * 2;
    const float *W = ptr; ptr += (long)k * C;
    const float *b = ptr;
    long expected = 20 + 4L * ((long)n_cat * 256 + (long)n_ord * 2 + (long)k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }
        int fidx = 0;
        for (int c = 0; c < n_cat; c++) {
            feat[fidx++] = cat_tables[(long)c * 256 + xi[c]];
        }
        for (int o = 0; o < n_ord; o++) {
            float v = xi[n_cat + o];
            float mean = ord_stats[(long)o * 2];
            float std = ord_stats[(long)o * 2 + 1];
            feat[fidx++] = (v - mean) / std;
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < k; j++) {
            const float *w = W + (long)j * C;
            float f = feat[j];
            for (int c = 0; c < C; c++) scores[c] += f * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```