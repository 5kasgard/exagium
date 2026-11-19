```hypothesis
Family: label-quadratic (factorized polynomial degree-2). The 8x8 optdigits pixels have strong local correlations; a full quadratic term would need 2080 features × 10 classes = 20k weights, but a rank-R factorization (R=4) reduces this to ~5k extra weights while capturing pairwise interactions. Training uses the same Adam recipe as label-linear on the expanded feature space (normalized pixels + factorized quadratic). Inference cost: ~6k FLOPs/item (vs 0.6k for linear), still far below GBDT/MLP. If dev accuracy ≤ 0.9613 (label-linear), the factorization is too restrictive or interactions don't help.
```

```python
"""label-quadratic: linear + factorized quadratic (rank R) on z-scored ordinal bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic 0x51554144 ('QUAD'), int32 d, R, C
  float32 mu[d], std[d]              # normalization
  float32 T[d][256][C]               # linear lookup: byte -> class scores
  float32 b1[C]                      # linear bias
  float32 U[C][R][d]                 # quadratic U factors
  float32 V[C][R][d]                 # quadratic V factors
"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
R_CANDIDATES = [2, 4, 8]          # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_grad(logits, y):
    """logits: (B, C), y: (B,) -> gradient (B, C)"""
    maxv = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - maxv)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def evaluate(X, y, params, d, C, R):
    """params: (T, b1, U, V, mu, std) -> accuracy"""
    T, b1, U, V, mu, std = params
    Z = (X - mu) / std                        # (n, d)
    lin = Z @ T.reshape(d, C) + b1            # (n, C)
    if R:
        # quadratic: sum_r (Z @ U_cr) * (Z @ V_cr)
        quad = np.zeros((X.shape[0], C), np.float32)
        for c in range(C):
            for r in range(R):
                u = U[c, r]
                v = V[c, r]
                quad[:, c] += (Z @ u) * (Z @ v)
        logits = lin + quad
    else:
        logits = lin
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_split(X_tr, y_tr, X_val, y_val, d, C, R, rng):
    """Train on X_tr, validate on X_val, return (params, val_acc)."""
    n = X_tr.shape[0]
    # normalization from training split
    mu = X_tr.mean(axis=0).astype(np.float32)
    std = X_tr.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z_tr = (X_tr - mu) / std
    Z_val = (X_val - mu) / std

    # linear lookup table T: for each byte value 0..255, we will store the contribution to each class.
    # But here we train on normalized features directly. We'll learn a linear weight matrix W_lin (d, C)
    # and then bake into T after training. During training we use Z.
    W_lin = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    U = (rng.uniform(-1, 1, (C, R, d)) * math.sqrt(6.0 / d)).astype(np.float32) if R else None
    V = (rng.uniform(-1, 1, (C, R, d)) * math.sqrt(6.0 / d)).astype(np.float32) if R else None

    params_list = [W_lin, b1]
    if R:
        params_list += [U, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params_list]

    steps_per_epoch = math.ceil(n / BATCH)
    total_steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n)) * steps_per_epoch
    t = 0
    best_val = -1.0
    best_params = None

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z_tr[idx]
            yy = y_tr[idx]
            bsz = len(yy)

            # forward
            lin = z @ W_lin + b1
            if R:
                quad = np.zeros((bsz, C), np.float32)
                for c in range(C):
                    for r in range(R):
                        u = U[c, r]
                        v = V[c, r]
                        quad[:, c] += (z @ u) * (z @ v)
                logits = lin + quad
            else:
                logits = lin

            grad = softmax_grad(logits, yy)                 # (bsz, C)

            # gradients
            g_W = z.T @ grad + L2 * W_lin
            g_b = grad.sum(axis=0)
            grads = [g_W, g_b]
            if R:
                g_U = np.zeros_like(U)
                g_V = np.zeros_like(V)
                for c in range(C):
                    gc = grad[:, c:c+1]                     # (bsz, 1)
                    for r in range(R):
                        u = U[c, r]
                        v = V[c, r]
                        zu = (z @ u)[:, None]               # (bsz, 1)
                        zv = (z @ v)[:, None]
                        g_U[c, r] = (gc * zv * z).sum(axis=0) + L2 * u
                        g_V[c, r] = (gc * zu * z).sum(axis=0) + L2 * v
                grads += [g_U, g_V]

            t += 1
            for i, (p, gr) in enumerate(zip(params_list, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation
        val_acc = evaluate(X_val, y_val, (W_lin, b1, U, V, mu, std), d, C, R)
        if val_acc > best_val:
            best_val = val_acc
            best_params = (W_lin.copy(), b1.copy(),
                           U.copy() if R else None,
                           V.copy() if R else None,
                           mu.copy(), std.copy())

    return best_params, best_val


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"]).astype(np.float32)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_R = 0
    best_acc = -1.0
    best_params = None

    for R in R_CANDIDATES:
        params, acc = train_split(X_tr, y_tr, X_val, y_val, d, C, R, rng)
        if acc > best_acc:
            best_acc = acc
            best_R = R
            best_params = params

    # retrain on full data with best_R
    rng_full = np.random.default_rng(SEED)
    full_params, _ = train_split(X, y, X, y, d, C, best_R, rng_full)
    W_lin, b1, U, V, mu, std = full_params

    # bake linear weights into lookup table T[d][256][C]
    # For each byte position j, for each possible byte value v (0..255), compute normalized value (v - mu[j])/std[j] * W_lin[j]
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        z = (np.arange(256, dtype=np.float32) - mu[j]) / std[j]
        T[j] = z[:, None] * W_lin[j:j+1]      # (256, C)

    # write weights
    magic = 0x51554144  # 'QUAD'
    R = best_R
    out = struct.pack("<4i", magic, d, R, C)
    out += mu.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if R:
        out += U.astype("<f4").tobytes()
        out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-quadratic inference: ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   uint32 magic=0x51554144, int32 d, R, C
 *   float32 mu[d], std[d]
 *   float32 T[d][256][C]   // linear contribution per byte value
 *   float32 b1[C]
 *   float32 U[C][R][d]     // quadratic factors
 *   float32 V[C][R][d]
 * Input record: d bytes (pixel counts 0..16). Output: 1 byte class index.
 */
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
    if (nw < 16 || hd[0] != 0x51554144) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mu = ptr; ptr += d;
    float *std = ptr; ptr += d;
    float *T = ptr; ptr += (long)d * 256 * C;
    float *b1 = ptr; ptr += C;
    float *U = R ? ptr : NULL; if (R) ptr += (long)C * R * d;
    float *V = R ? ptr : NULL; if (R) ptr += (long)C * R * d;

    long expected = 16 + 4L * (2L * d + (long)d * 256 * C + C + (R ? 2L * C * R * d : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / std[j];
        }

        // linear part via lookup table T
        for (int c = 0; c < C; c++) score[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) score[c] += tj[c];
        }

        // quadratic part
        if (R) {
            for (int c = 0; c < C; c++) {
                float quad = 0.0f;
                for (int r = 0; r < R; r++) {
                    const float *uc = U + ((long)c * R + r) * d;
                    const float *vc = V + ((long)c * R + r) * d;
                    float dotu = 0.0f, dotv = 0.0f;
                    for (int j = 0; j < d; j++) {
                        dotu += z[j] * uc[j];
                        dotv += z[j] * vc[j];
                    }
                    quad += dotu * dotv;
                }
                score[c] += quad;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```