```hypothesis
The label-poly2 family extends label-linear by adding squared z-scored features for each ordinal pixel. Handwritten digit intensities have non-linear relationships (e.g., stroke thickness vs. background) that a linear model cannot capture; quadratic terms model this curvature while keeping inference identical (same table lookup, same cost). If dev accuracy does not exceed label-linear's 0.9613, the extra capacity does not help here.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (same as label-linear)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encodings(X, types, use_quad):
    """Return list of encoding matrices E_j (256 x dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            z = (np.arange(256) - mu) / sd
            if use_quad:
                E = np.column_stack([z, z * z]).astype(np.float32)   # 256 x 2
            else:
                E = z[:, None].astype(np.float32)                     # 256 x 1
        else:   # categorical (not used in this task, but kept for completeness)
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    return enc

def train_model(X, y, enc, C, rng):
    """Train logistic regression on encoded features, return W1 (k x C), b1 (C)."""
    n = X.shape[0]
    k = sum(E.shape[1] for E in enc)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    W1 = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(S):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        return P

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W1 + b1
            P = sg(logits)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)

            grad_W1 = z.T @ g + L2 * W1
            grad_b1 = g.sum(0)
            grads = [grad_W1, grad_b1]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W1, b1

def evaluate(X, y, enc, W1, b1):
    """Compute accuracy on given data."""
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]

    # Split: last 20% for tuning (file order)
    n = len(X_all)
    tune_size = n // 5
    train_X, tune_X = X_all[:-tune_size], X_all[-tune_size:]
    train_y, tune_y = y_all[:-tune_size], y_all[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_quad, best_W1, best_b1, best_enc = -1, False, None, None, None

    for use_quad in (False, True):
        enc = build_encodings(train_X, types, use_quad)
        W1, b1 = train_model(train_X, train_y, enc, C, np.random.default_rng(SEED))
        acc = evaluate(tune_X, tune_y, enc, W1, b1)
        if acc > best_acc:
            best_acc, best_quad = acc, use_quad
            best_W1, best_b1, best_enc = W1, b1, enc

    # Retrain on full data with best setting
    final_enc = build_encodings(X_all, types, best_quad)
    final_W1, final_b1 = train_model(X_all, y_all, final_enc, C, np.random.default_rng(SEED))

    # Compute per-feature tables T_j = E_j @ W1_j
    off = 0
    T_list = []
    for E in final_enc:
        dim = E.shape[1]
        T_list.append((E @ final_W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)   # (d, 256, C)

    # Write weights.bin: magic, d, H=0, C, then T (d*256*C floats), b1 (C floats)
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += final_b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```