```hypothesis
Family: label-fm (Factorization Machine). Optdigits pixels exhibit strong pairwise spatial correlations (strokes, loops) that a linear model misses. A factorization machine models all pairwise interactions via a low-rank factorization (k=8 latent factors), adding only ~5k parameters over the linear baseline. Inference computes scores in O(d·k·C) ≈ 5000 FLOPs (d=64, C=10), using only adds/multiplies—no exponentials or branching—so it should be faster than label-mlp16 while matching its accuracy. Raw pixel values 0..16 work directly without normalization. Failure: dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 10000.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    idx_tr, idx_val = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X[idx_tr].astype(np.float32), y[idx_tr]
    X_val, y_val = X[idx_val].astype(np.float32), y[idx_val]

    rng = np.random.default_rng(SEED)

    def init_params():
        w0 = np.zeros(C, np.float32)
        W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
        V = (rng.uniform(-1, 1, (d, K, C)) * math.sqrt(6.0 / (d * K))).astype(np.float32)
        return [w0, W, V]

    def compute_scores(Xb, params):
        w0, W, V = params
        linear = Xb @ W + w0
        inter = np.zeros((Xb.shape[0], C), np.float32)
        for c in range(C):
            for f in range(K):
                v_cf = V[:, f, c]
                S = Xb @ v_cf
                S2 = (Xb * Xb) @ (v_cf * v_cf)
                inter[:, c] += 0.5 * (S * S - S2)
        return linear + inter

    def compute_grads(Xb, yb, params):
        w0, W, V = params
        B = Xb.shape[0]
        scores = compute_scores(Xb, params)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        grad_scores = probs
        grad_scores[np.arange(B), yb] -= 1.0
        grad_scores /= B

        gw0 = grad_scores.sum(axis=0)
        gW = Xb.T @ grad_scores + 2 * L2 * W
        gV = np.zeros((d, K, C), np.float32)
        Xb2 = Xb * Xb
        for c in range(C):
            g_c = grad_scores[:, c]
            for f in range(K):
                v_cf = V[:, f, c]
                S = Xb @ v_cf
                term1 = Xb.T @ (g_c[:, None] * S[:, None])
                term2 = v_cf * (Xb2.T @ g_c)
                gV[:, f, c] = term1 - term2 + 2 * L2 * v_cf
        return [gw0, gW, gV]

    # Train on tuning split to pick best epoch
    params = init_params()
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0
    best_val_acc, best_epoch, best_params = -1, 0, None
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))

    for epoch in range(max_epochs):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s + BATCH]
            grads = compute_grads(X_tr[idx], y_tr[idx], params)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                state_m[i] *= 0.9; state_m[i] += 0.1 * gr
                state_v[i] *= 0.999; state_v[i] += 0.001 * gr * gr
                m_hat = state_m[i] / (1 - 0.9 ** t)
                v_hat = state_v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        val_scores = compute_scores(X_val, params)
        val_acc = (val_scores.argmax(axis=1) == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_params = [p.copy() for p in params]

    # Retrain on full data for best_epoch+1 epochs
    X_full, y_full = X.astype(np.float32), y
    params = init_params()
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            grads = compute_grads(X_full[idx], y_full[idx], params)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                state_m[i] *= 0.9; state_m[i] += 0.1 * gr
                state_v[i] *= 0.999; state_v[i] += 0.001 * gr * gr
                m_hat = state_m[i] / (1 - 0.9 ** t)
                v_hat = state_v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    w0, W, V = params
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (C + (long)d * C + (long)d * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *W = w0 + C;
    const float *V = W + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vf = malloc(sizeof(float) * K);
    float *sum_vf2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        unsigned char best = 0;
        float best_score = -1e30f;

        for (int c = 0; c < C; c++) {
            float score = w0[c];
            for (int j = 0; j < d; j++) score += W[(long)j * C + c] * xi[j];

            for (int f = 0; f < K; f++) { sum_vf[f] = 0.0f; sum_vf2[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                float xj = xi[j], xj2 = xj * xj;
                for (int f = 0; f < K; f++) {
                    float v = V[((long)j * K + f) * C + c];
                    sum_vf[f] += v * xj;
                    sum_vf2[f] += v * v * xj2;
                }
            }
            for (int f = 0; f < K; f++) score += 0.5f * (sum_vf[f] * sum_vf[f] - sum_vf2[f]);

            if (c == 0 || score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```