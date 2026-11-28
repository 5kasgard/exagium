```hypothesis
Family: label-adjpoly (Adjacent Polynomial). Adds pairwise products of horizontally/vertically adjacent pixels (112 pairs) to a linear model, capturing local stroke/corner patterns in 8×8 digits. Only 1770 parameters; inference ~2.7× linear cost. Fails if dev accuracy ≤ 0.9613 (label-linear) or CPU cost > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
L2_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_GRID = [1e-3, 3e-3, 1e-2]
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 64

    # Adjacent pairs for 8x8 grid
    pairs = []
    for r in range(8):
        for c in range(7):
            pairs.append((r * 8 + c, r * 8 + c + 1))
    for r in range(7):
        for c in range(8):
            pairs.append((r * 8 + c, (r + 1) * 8 + c))
    n_pairs = len(pairs)  # 112

    # Split: last 20% in file order for tuning
    split = int(n * 0.8)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Per-pixel normalization on train
    means = X_train.mean(axis=0).astype(np.float32)
    stds = X_train.std(axis=0).astype(np.float32)
    stds[stds == 0] = 1.0

    def normalize(X_raw):
        return (X_raw.astype(np.float32) - means) / stds

    def build_pair_feat(X_norm):
        n = X_norm.shape[0]
        pair_feat = np.zeros((n, n_pairs), dtype=np.float32)
        for p, (i, j) in enumerate(pairs):
            pair_feat[:, p] = X_norm[:, i] * X_norm[:, j]
        return pair_feat

    X_train_norm = normalize(X_train)
    X_tune_norm = normalize(X_tune)
    X_train_pair = build_pair_feat(X_train_norm)
    X_tune_pair = build_pair_feat(X_tune_norm)

    best_acc = -1.0
    best_hp = None
    rng = np.random.default_rng(SEED)

    for L2 in L2_GRID:
        for LR in LR_GRID:
            W_lin = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
            W_pair = (rng.uniform(-1, 1, (n_pairs, C)) * math.sqrt(6.0 / n_pairs)).astype(np.float32)
            b = np.zeros(C, np.float32)
            params = [W_lin, W_pair, b]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0

            n_train = X_train_norm.shape[0]
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
            for _ in range(epochs):
                order = rng.permutation(n_train)
                for s in range(0, n_train, BATCH):
                    idx_b = order[s:s + BATCH]
                    x_lin = X_train_norm[idx_b]
                    x_pair = X_train_pair[idx_b]
                    yy = y_train[idx_b]

                    logits = x_lin @ W_lin + x_pair @ W_pair + b
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yy)), yy] -= 1.0
                    grad_logits = (probs / len(yy)).astype(np.float32)

                    g_W_lin = x_lin.T @ grad_logits + L2 * W_lin
                    g_W_pair = x_pair.T @ grad_logits + L2 * W_pair
                    g_b = grad_logits.sum(axis=0)
                    grads = [g_W_lin, g_W_pair, g_b]
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            logits_tune = X_tune_norm @ W_lin + X_tune_pair @ W_pair + b
            acc = (logits_tune.argmax(axis=1) == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_hp = (L2, LR)

    # Retrain on full data with best hyperparameters
    L2, LR = best_hp
    X_full_norm = normalize(X)
    X_full_pair = build_pair_feat(X_full_norm)

    W_lin = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    W_pair = (rng.uniform(-1, 1, (n_pairs, C)) * math.sqrt(6.0 / n_pairs)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W_lin, W_pair, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n_full = X_full_norm.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx_b = order[s:s + BATCH]
            x_lin = X_full_norm[idx_b]
            x_pair = X_full_pair[idx_b]
            yy = y[idx_b]

            logits = x_lin @ W_lin + x_pair @ W_pair + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)

            g_W_lin = x_lin.T @ grad_logits + L2 * W_lin
            g_W_pair = x_pair.T @ grad_logits + L2 * W_pair
            g_b = grad_logits.sum(axis=0)
            grads = [g_W_lin, g_W_pair, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic, d, n_pairs, C, means[d], stds[d], W_lin[d*C], W_pair[n_pairs*C], b[C]
    magic = 0x41444A50  # 'PJDA' little-endian
    out = struct.pack("<4i", magic, d, n_pairs, C)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += W_pair.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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

static const int pair_i[112] = {
    0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7,
    8,9, 9,10, 10,11, 11,12, 12,13, 13,14, 14,15,
    16,17, 17,18, 18,19, 19,20, 20,21, 21,22, 22,23,
    24,25, 25,26, 26,27, 27,28, 28,29, 29,30, 30,31,
    32,33, 33,34, 34,35, 35,36, 36,37, 37,38, 38,39,
    40,41, 41,42, 42,43, 43,44, 44,45, 45,46, 46,47,
    48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55,
    56,57, 57,58, 58,59, 59,60, 60,61, 61,62, 62,63,
    0,8, 1,9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15,
    8,16, 9,17, 10,18, 11,19, 12,20, 13,21, 14,22, 15,23,
    16,24, 17,25, 18,26, 19,27, 20,28, 21,29, 22,30, 23,31,
    24,32, 25,33, 26,34, 27,35, 28,36, 29,37, 30,38, 31,39,
    32,40, 33,41, 34,42, 35,43, 36,44, 37,45, 38,46, 39,47,
    40,48, 41,49, 42,50, 43,51, 44,52, 45,53, 46,54, 47,55,
    48,56, 49,57, 50,58, 51,59, 52,60, 53,61, 54,62, 55,63
};
static const int pair_j[112] = {
    1,2,3,4,5,6,7,
    9,10,11,12,13,14,15,
    17,18,19,20,21,22,23,
    25,26,27,28,29,30,31,
    33,34,35,36,37,38,39,
    41,42,43,44,45,46,47,
    49,50,51,52,53,54,55,
    57,58,59,60,61,62,63,
    8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,
    24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,
    40,41,42,43,44,45,46,47,
    48,49,50,51,52,53,54,55,
    56,57,58,59,60,61,62,63
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x41444A50) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_pairs = hd[2], C = hd[3];
    if (d != 64 || n_pairs != 112 || C != 10) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * (d + d + d * C + n_pairs * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *means = (float *)(hd + 4);
    float *stds = means + d;
    float *W_lin = stds + d;
    float *W_pair = W_lin + d * C;
    float *bias = W_pair + n_pairs * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_norm = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            x_norm[j] = (xi[j] - means[j]) / stds[j];
        }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float xj = x_norm[j];
            if (xj == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += xj * W_lin[j * C + c];
        }
        for (int p = 0; p < n_pairs; p++) {
            float prod = x_norm[pair_i[p]] * x_norm[pair_j[p]];
            if (prod == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += prod * W_pair[p * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```