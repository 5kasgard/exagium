```hypothesis
Family: label-fm (Factorization Machine with low-rank pairwise interactions).
Why: Optdigits pixels have strong local correlations (strokes); a linear model on z-scored pixels (label-linear, 96.1%) cannot capture these. FM adds all pairwise interactions via a rank-K factorization (cost ~640 + 138K mul-adds/item), capturing stroke structure at a fraction of an MLP's cost. Tuning K ∈ {4,8,16} on the last 20% of train should yield >97% dev accuracy at ~2000 Ir/item.
Failure signal: If dev accuracy ≤ 96.1% (label-linear) or Ir/item > 5000, the interactions are not helping or the optimization is ineffective.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_eval(K, X_tr, y_tr, X_val, y_val, d, C, mu, sd):
    rng = np.random.default_rng(SEED)
    W_lin = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K)) * math.sqrt(6.0 / d)).astype(np.float32)
    W_int = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W_lin, V, W_int, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    X_tr_z = ((X_tr.astype(np.float32) - mu) / sd)
    X_val_z = ((X_val.astype(np.float32) - mu) / sd)
    n_tr = X_tr.shape[0]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            x, yy = X_tr_z[idx], y_tr[idx]

            lin = x @ W_lin + b
            s_k = x @ V
            ss_k = (x * x) @ (V * V)
            inter = 0.5 * (s_k * s_k - ss_k)
            logits = lin + inter @ W_int

            g = softmax_cross_entropy_grad(logits, yy)

            g_W_lin = x.T @ g + L2 * W_lin
            g_b = g.sum(0) + L2 * b
            g_W_int = inter.T @ g + L2 * W_int

            g_inter = g @ W_int.T
            g_V = np.zeros_like(V)
            for k_idx in range(K):
                factor = g_inter[:, k_idx] * (s_k[:, k_idx] - x * V[:, k_idx])
                g_V[:, k_idx] = x.T @ factor
            g_V += L2 * V

            grads = [g_W_lin, g_V, g_W_int, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # validation accuracy
    x = X_val_z
    lin = x @ W_lin + b
    s_k = x @ V
    ss_k = (x * x) @ (V * V)
    inter = 0.5 * (s_k * s_k - ss_k)
    logits = lin + inter @ W_int
    preds = logits.argmax(axis=1)
    return (preds == y_val).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    split = int(n * 0.8)
    X_tr, X_val = X_bytes[:split], X_bytes[split:]
    y_tr, y_val = y[:split], y[split:]

    mu = X_tr.mean(axis=0).astype(np.float32)
    sd = X_tr.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    best_K, best_acc = 0, -1.0
    for K in (4, 8, 16):
        acc = train_eval(K, X_tr, y_tr, X_val, y_val, d, C, mu, sd)
        print(f"K={K} val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # Retrain on all data with best_K
    K = best_K
    rng = np.random.default_rng(SEED)
    W_lin = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K)) * math.sqrt(6.0 / d)).astype(np.float32)
    W_int = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W_lin, V, W_int, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_all_z = ((X_bytes.astype(np.float32) - mu) / sd)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x, yy = X_all_z[idx], y[idx]

            lin = x @ W_lin + b
            s_k = x @ V
            ss_k = (x * x) @ (V * V)
            inter = 0.5 * (s_k * s_k - ss_k)
            logits = lin + inter @ W_int

            g = softmax_cross_entropy_grad(logits, yy)

            g_W_lin = x.T @ g + L2 * W_lin
            g_b = g.sum(0) + L2 * b
            g_W_int = inter.T @ g + L2 * W_int

            g_inter = g @ W_int.T
            g_V = np.zeros_like(V)
            for k_idx in range(K):
                factor = g_inter[:, k_idx] * (s_k[:, k_idx] - x * V[:, k_idx])
                g_V[:, k_idx] = x.T @ factor
            g_V += L2 * V

            grads = [g_W_lin, g_V, g_W_int, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x464D4C42  # "FMLB"
    out = struct.pack("<4i", magic, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W_int.astype("<f4").tobytes()
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K < 1 || K > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (2 * d + d * C + d * K + K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mu = ptr; ptr += d;
    float *sd = ptr; ptr += d;
    float *W_lin = ptr; ptr += d * C;
    float *V = ptr; ptr += d * K;
    float *W_int = ptr; ptr += K * C;
    float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *s_k = malloc(K * sizeof(float));
    float *ss_k = malloc(K * sizeof(float));
    float *inter = malloc(K * sizeof(float));
    float *x_z = malloc(d * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            x_z[j] = (xi[j] - mu[j]) / sd[j];
        }

        for (int c = 0; c < C; c++) {
            scores[c] = b[c];
            for (int j = 0; j < d; j++) {
                scores[c] += x_z[j] * W_lin[j * C + c];
            }
        }

        for (int k_idx = 0; k_idx < K; k_idx++) {
            s_k[k_idx] = 0.0f;
            ss_k[k_idx] = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[j * K + k_idx];
                float xv = x_z[j] * v;
                s_k[k_idx] += xv;
                ss_k[k_idx] += xv * xv;
            }
            inter[k_idx] = 0.5f * (s_k[k_idx] * s_k[k_idx] - ss_k[k_idx]);
        }

        for (int c = 0; c < C; c++) {
            for (int k_idx = 0; k_idx < K; k_idx++) {
                scores[c] += inter[k_idx] * W_int[k_idx * C + c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```