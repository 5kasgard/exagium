```hypothesis
Family: label-fm (factorization machine with degree-2 interactions).
Why: Optdigits pixels have local correlations; a low-rank quadratic model captures pairwise interactions cheaply. It adds only O(d*k + k*C) parameters and O(d*k + k*C) ops per item (k=10), staying near label-linear cost while modeling non-linearities.
Failure sign: Dev accuracy ≤ label-linear (0.9613) or instructions/item > 2× label-linear.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
K_OPTIONS = [5, 10, 15, 20]
LR_OPTIONS = [1e-3, 3e-3, 1e-2]
L2_OPTIONS = [1e-4, 1e-3]
EPOCH_OPTIONS = [20, 50, 100]

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
    d = X.shape[1]
    n = X.shape[0]
    val_size = n // 5
    tr_idx = np.arange(n - val_size)
    va_idx = np.arange(n - val_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    mu_tr = X_tr.mean(axis=0).astype(np.float64)
    sd_tr = X_tr.std(axis=0).astype(np.float64)
    sd_tr[sd_tr == 0] = 1.0
    def norm(X, mu, sd): return (X.astype(np.float64) - mu) / sd
    X_tr_n = norm(X_tr, mu_tr, sd_tr).astype(np.float32)
    X_va_n = norm(X_va, mu_tr, sd_tr).astype(np.float32)

    best_acc = -1.0
    best_cfg = None
    best_weights = None
    rng = np.random.default_rng(SEED)

    for k in K_OPTIONS:
        for lr in LR_OPTIONS:
            for l2 in L2_OPTIONS:
                for max_ep in EPOCH_OPTIONS:
                    W_lin = rng.normal(0, 0.01, (d, C)).astype(np.float32)
                    b_lin = np.zeros(C, np.float32)
                    V = rng.normal(0, 0.01, (d, k)).astype(np.float32)
                    U = rng.normal(0, 0.01, (k, C)).astype(np.float32)
                    params = [W_lin, b_lin, V, U]
                    m = [np.zeros_like(p) for p in params]
                    v = [np.zeros_like(p) for p in params]
                    t = 0
                    n_tr = X_tr_n.shape[0]
                    for _ in range(max_ep):
                        order = rng.permutation(n_tr)
                        for s in range(0, n_tr, BATCH):
                            idx = order[s:s+BATCH]
                            x = X_tr_n[idx]
                            yy = y_tr[idx]
                            z = x @ V
                            z2 = z * z
                            scores = x @ W_lin + b_lin + z2 @ U
                            mx = scores.max(axis=1, keepdims=True)
                            ex = np.exp(scores - mx)
                            probs = ex / ex.sum(axis=1, keepdims=True)
                            probs[np.arange(len(yy)), yy] -= 1.0
                            probs /= len(yy)
                            gW = x.T @ probs + l2 * W_lin
                            gb = probs.sum(axis=0)
                            gU = z2.T @ probs + l2 * U
                            dL_dz2 = probs @ U.T
                            dL_dz = 2 * z * dL_dz2
                            gV = x.T @ dL_dz + l2 * V
                            grads = [gW, gb, gV, gU]
                            t += 1
                            for i, (p, gr) in enumerate(zip(params, grads)):
                                m[i] = 0.9 * m[i] + 0.1 * gr
                                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                                mh = m[i] / (1 - 0.9 ** t)
                                vh = v[i] / (1 - 0.999 ** t)
                                p -= lr * mh / (np.sqrt(vh) + 1e-8)
                    z_va = X_va_n @ V
                    z2_va = z_va * z_va
                    sc_va = X_va_n @ W_lin + b_lin + z2_va @ U
                    acc = (sc_va.argmax(axis=1) == y_va).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (k, lr, l2, max_ep)
                        best_weights = (W_lin.copy(), b_lin.copy(), V.copy(), U.copy())

    print(f"Best val acc: {best_acc:.4f}, cfg: {best_cfg}", file=sys.stderr)
    k, lr, l2, max_ep = best_cfg

    # Retrain on full data
    rng = np.random.default_rng(SEED)
    W_lin = rng.normal(0, 0.01, (d, C)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    V = rng.normal(0, 0.01, (d, k)).astype(np.float32)
    U = rng.normal(0, 0.01, (k, C)).astype(np.float32)
    params = [W_lin, b_lin, V, U]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    mu_full = X.mean(axis=0).astype(np.float64)
    sd_full = X.std(axis=0).astype(np.float64)
    sd_full[sd_full == 0] = 1.0
    X_full_n = norm(X, mu_full, sd_full).astype(np.float32)
    n_full = X_full_n.shape[0]
    for _ in range(max_ep):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s+BATCH]
            x = X_full_n[idx]
            yy = y[idx]
            z = x @ V
            z2 = z * z
            scores = x @ W_lin + b_lin + z2 @ U
            mx = scores.max(axis=1, keepdims=True)
            ex = np.exp(scores - mx)
            probs = ex / ex.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)
            gW = x.T @ probs + l2 * W_lin
            gb = probs.sum(axis=0)
            gU = z2.T @ probs + l2 * U
            dL_dz2 = probs @ U.T
            dL_dz = 2 * z * dL_dz2
            gV = x.T @ dL_dz + l2 * V
            grads = [gW, gb, gV, gU]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                mh = m[i] / (1 - 0.9 ** t)
                vh = v[i] / (1 - 0.999 ** t)
                p -= lr * mh / (np.sqrt(vh) + 1e-8)

    # Build lookup tables
    norm_vals = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        for v in range(256):
            norm_vals[j, v] = (v - mu_full[j]) / sd_full[j]
    T_lin = np.zeros((d, 256, C), dtype=np.float32)
    T_quad = np.zeros((d, 256, k), dtype=np.float32)
    for j in range(d):
        for v in range(256):
            nv = norm_vals[j, v]
            T_lin[j, v, :] = W_lin[j, :] * nv
            T_quad[j, v, :] = V[j, :] * nv

    magic = 0x464D5032  # "FM2"
    out = struct.pack("<4i", magic, d, k, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_quad.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x464D5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * k + C + (long)k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_quad = T_lin + (long)d * 256 * C;
    const float *b_lin = T_quad + (long)d * 256 * k;
    const float *U = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = calloc(k, sizeof(float));
    float *score_lin = malloc(C * sizeof(float));
    float *score_quad = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        memcpy(score_lin, b_lin, C * sizeof(float));
        memset(z, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *tl = T_lin + ((long)j * 256 + xi[j]) * C;
            const float *tq = T_quad + ((long)j * 256 + xi[j]) * k;
            for (int c = 0; c < C; c++) score_lin[c] += tl[c];
            for (int f = 0; f < k; f++) z[f] += tq[f];
        }
        memset(score_quad, 0, C * sizeof(float));
        for (int f = 0; f < k; f++) {
            float z2 = z[f] * z[f];
            const float *uf = U + (long)f * C;
            for (int c = 0; c < C; c++) score_quad[c] += uf[c] * z2;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (score_lin[c] + score_quad[c] > score_lin[best] + score_quad[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```