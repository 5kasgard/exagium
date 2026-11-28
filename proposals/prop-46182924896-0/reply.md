```hypothesis
Family: label-interact (logistic regression with target-encoded categoricals + ordinal pairwise interactions).
Why: Adult has 6 ordinal and 8 categorical features; target encoding compresses high-cardinality categoricals into single informative dimensions while pairwise ordinal products capture non-linear numeric effects (e.g., age×hours). This yields a 29-feature linear model richer than label-linear's one-hot but far smaller/faster than label-mlp16.
Failure sign: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 1000 Ir/item.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
ALPHA = 10.0

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    d = card["input"]["bytes"]
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    n = X.shape[0]

    types = card["input"]["types"]
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord = len(ord_cols)
    n_cat = len(cat_cols)
    n_pair = n_ord * (n_ord - 1) // 2
    n_feat = n_ord + n_cat + n_pair

    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    ord_mean = np.zeros(n_ord, np.float64)
    ord_std = np.zeros(n_ord, np.float64)
    for i, col in enumerate(ord_cols):
        v = X[idx_train, col].astype(np.float64)
        ord_mean[i] = v.mean()
        ord_std[i] = v.std() or 1.0

    global_mean = y[idx_train].mean()
    cat_enc = np.zeros((n_cat, 256), np.float32)
    for i, col in enumerate(cat_cols):
        vals = X[idx_train, col]
        y_tr = y[idx_train]
        for val in range(256):
            mask = (vals == val)
            cnt = mask.sum()
            if cnt > 0:
                pos = y_tr[mask].sum()
                cat_enc[i, val] = (pos + ALPHA * global_mean) / (cnt + ALPHA)
            else:
                cat_enc[i, val] = global_mean
        cat_enc[i, 255] = global_mean

    def build_features(indices):
        m = len(indices)
        Z = np.zeros((m, n_feat), np.float32)
        for i, col in enumerate(ord_cols):
            Z[:, i] = (X[indices, col].astype(np.float32) - ord_mean[i]) / ord_std[i]
        for i, col in enumerate(cat_cols):
            Z[:, n_ord + i] = cat_enc[i, X[indices, col]]
        k = n_ord + n_cat
        for i in range(n_ord):
            for j in range(i + 1, n_ord):
                Z[:, k] = Z[:, i] * Z[:, j]
                k += 1
        return Z

    Z_train = build_features(idx_train)
    Z_tune = build_features(idx_tune)
    y_train = y[idx_train]
    y_tune = y[idx_tune]

    best_acc = -1.0
    best_lr = 0.003
    best_l2 = 1e-4
    rng = np.random.default_rng(SEED)

    for lr in [0.001, 0.003, 0.01]:
        for l2 in [0.0001, 0.001, 0.01]:
            W = (rng.uniform(-1, 1, (n_feat, 2)) * math.sqrt(6.0 / n_feat)).astype(np.float32)
            b = np.zeros(2, np.float32)
            params = [W, b]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            max_epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / split)))
            for _ in range(max_epochs):
                order = rng.permutation(split)
                for s in range(0, split, BATCH):
                    idx = order[s:s + BATCH]
                    z = Z_train[idx]
                    yy = y_train[idx]
                    logits = z @ W + b
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    grad_logits = probs
                    grad_logits[np.arange(len(yy)), yy] -= 1.0
                    grad_logits /= len(yy)
                    grad_W = z.T @ grad_logits + l2 * W
                    grad_b = grad_logits.sum(0)
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            logits_tune = Z_tune @ params[0] + params[1]
            pred = logits_tune.argmax(axis=1)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_lr, best_l2 = lr, l2

    print(f"Best tune acc: {best_acc:.4f} lr={best_lr} l2={best_l2}", file=sys.stderr)

    ord_mean_full = np.zeros(n_ord, np.float64)
    ord_std_full = np.zeros(n_ord, np.float64)
    for i, col in enumerate(ord_cols):
        v = X[:, col].astype(np.float64)
        ord_mean_full[i] = v.mean()
        ord_std_full[i] = v.std() or 1.0

    global_mean_full = y.mean()
    cat_enc_full = np.zeros((n_cat, 256), np.float32)
    for i, col in enumerate(cat_cols):
        vals = X[:, col]
        for val in range(256):
            mask = (vals == val)
            cnt = mask.sum()
            if cnt > 0:
                pos = y[mask].sum()
                cat_enc_full[i, val] = (pos + ALPHA * global_mean_full) / (cnt + ALPHA)
            else:
                cat_enc_full[i, val] = global_mean_full
        cat_enc_full[i, 255] = global_mean_full

    Z_full = np.zeros((n, n_feat), np.float32)
    for i, col in enumerate(ord_cols):
        Z_full[:, i] = (X[:, col].astype(np.float32) - ord_mean_full[i]) / ord_std_full[i]
    for i, col in enumerate(cat_cols):
        Z_full[:, n_ord + i] = cat_enc_full[i, X[:, col]]
    k = n_ord + n_cat
    for i in range(n_ord):
        for j in range(i + 1, n_ord):
            Z_full[:, k] = Z_full[:, i] * Z_full[:, j]
            k += 1

    lr, l2 = best_lr, best_l2
    W = (rng.uniform(-1, 1, (n_feat, 2)) * math.sqrt(6.0 / n_feat)).astype(np.float32)
    b = np.zeros(2, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n)))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z_full[idx]
            yy = y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yy)), yy] -= 1.0
            grad_logits /= len(yy)
            grad_W = z.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(0)
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    col_type = np.array([0 if t == "ordinal" else 1 for t in types], dtype=np.uint8)
    W_final, b_final = params

    out = struct.pack("<5i", 0x494E5452, d, n_ord, n_cat, n_feat)
    out += col_type.tobytes()
    out += ord_mean_full.astype(np.float32).tobytes()
    out += ord_std_full.astype(np.float32).tobytes()
    out += cat_enc_full.astype(np.float32).tobytes()
    out += W_final.astype(np.float32).tobytes()
    out += b_final.astype(np.float32).tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20 || hd[0] != 0x494E5452) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_ord = hd[2], n_cat = hd[3], n_feat = hd[4];
    int n_pair = n_ord * (n_ord - 1) / 2;
    if (n_feat != n_ord + n_cat + n_pair) { fprintf(stderr, "bad n_feat\n"); return 3; }

    uint8_t *col_type = (uint8_t *)(hd + 5);
    float *ord_mean = (float *)(col_type + d);
    float *ord_std = ord_mean + n_ord;
    float *cat_enc = ord_std + n_ord;
    float *W = cat_enc + (long)n_cat * 256;
    float *b = W + (long)n_feat * 2;

    long expected = 20 + d + 4L * (n_ord + n_ord + (long)n_cat * 256 + (long)n_feat * 2 + 2);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *ord_feat = malloc(sizeof(float) * n_ord);
    float *cat_feat = malloc(sizeof(float) * n_cat);
    float *pair_feat = malloc(sizeof(float) * n_pair);
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int ord_idx = 0, cat_idx = 0;
        for (int j = 0; j < d; j++) {
            if (col_type[j] == 0) {
                ord_feat[ord_idx] = (xi[j] - ord_mean[ord_idx]) / ord_std[ord_idx];
                ord_idx++;
            } else {
                cat_feat[cat_idx] = cat_enc[(long)cat_idx * 256 + xi[j]];
                cat_idx++;
            }
        }

        int p = 0;
        for (int u = 0; u < n_ord; u++) {
            for (int v = u + 1; v < n_ord; v++) {
                pair_feat[p++] = ord_feat[u] * ord_feat[v];
            }
        }

        scores[0] = b[0];
        scores[1] = b[1];

        for (int f = 0; f < n_ord; f++) {
            float val = ord_feat[f];
            scores[0] += val * W[f * 2 + 0];
            scores[1] += val * W[f * 2 + 1];
        }
        for (int f = 0; f < n_cat; f++) {
            float val = cat_feat[f];
            int wi = (n_ord + f) * 2;
            scores[0] += val * W[wi + 0];
            scores[1] += val * W[wi + 1];
        }
        for (int f = 0; f < n_pair; f++) {
            float val = pair_feat[f];
            int wi = (n_ord + n_cat + f) * 2;
            scores[0] += val * W[wi + 0];
            scores[1] += val * W[wi + 1];
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```