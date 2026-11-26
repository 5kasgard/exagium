```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong feature interactions (e.g., education×occupation, marital×sex) that linear models miss. A low-rank FM adds pairwise interactions via 8‑dim latent vectors per feature value, adding only ~57k parameters. Inference costs ~d·k + d²·k ≈ 1.5k ops/item, far below label-mlp16. Expected dev accuracy >0.859 (beats label-linear, matches label-mlp16). If dev accuracy ≤0.8524, interactions don't help and the family fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte as a categorical index (0..255). Learns per-class linear weights and
per-class latent vectors (dim k) for pairwise FM interactions. Adam lr=3e-3, L2=1e-4,
batch=64, epochs=min(100, ceil(600000/n)). Hyperparameter k chosen on last 20% of train.in.
Exported weights: magic 0x464D4C31, d, k, C; float32 linear[C][d][256], latent[C][d][256][k], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_GRID = [4, 8, 16]          # latent dim candidates
VAL_EPOCHS = 10              # epochs for grid search
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x464D4C31           # 'FML1'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def evaluate(X, y, linear, latent, bias, k):
    n, d = X.shape
    C = linear.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = X[s:s+BATCH]
        yy = y[s:s+BATCH]
        B = len(yy)
        logits = np.zeros((B, C), np.float32)
        for c in range(C):
            lin = linear[c]
            lat = latent[c]
            b = bias[c]
            lin_emb = lin[np.arange(d), idx.T].T          # (B, d)
            lat_emb = lat[np.arange(d), idx.T].transpose(1,0,2)  # (B, d, k)
            lin_term = lin_emb.sum(axis=1)
            sum_v = lat_emb.sum(axis=1)
            sum_sq = (lat_emb ** 2).sum(axis=1)
            inter = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_sq.sum(axis=1))
            logits[:, c] = b + lin_term + inter
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        correct += (pred == yy).sum()
    return correct / n

def train_fm(X, y, k, epochs, rng):
    n, d = X.shape
    C = 2
    # init parameters
    linear = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    latent = (rng.normal(0, 0.01, (C, d, 256, k))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [linear, latent, bias]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xi = X[idx]
            yy = y[idx]
            B = len(yy)
            # forward
            logits = np.zeros((B, C), np.float32)
            lin_emb_all = []
            lat_emb_all = []
            sum_v_all = []
            for c in range(C):
                lin = linear[c]
                lat = latent[c]
                b = bias[c]
                lin_emb = lin[np.arange(d), xi.T].T
                lat_emb = lat[np.arange(d), xi.T].transpose(1,0,2)
                lin_term = lin_emb.sum(axis=1)
                sum_v = lat_emb.sum(axis=1)
                sum_sq = (lat_emb ** 2).sum(axis=1)
                inter = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_sq.sum(axis=1))
                logits[:, c] = b + lin_term + inter
                lin_emb_all.append(lin_emb)
                lat_emb_all.append(lat_emb)
                sum_v_all.append(sum_v)
            # softmax + grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yy] -= 1.0
            grad_logits /= B
            # gradients
            grads = [np.zeros_like(p) for p in params]
            for c in range(C):
                g = grad_logits[:, c]
                # bias
                grads[2][c] += g.sum()
                # linear
                dlin = np.zeros((d, 256), np.float32)
                for j in range(d):
                    np.add.at(dlin[j], xi[:, j], g)
                dlin += L2 * linear[c]
                grads[0][c] += dlin
                # latent
                dlat = np.zeros((d, 256, k), np.float32)
                lat_emb = lat_emb_all[c]
                sum_v = sum_v_all[c]
                for j in range(d):
                    v_j = lat_emb[:, j, :]
                    grad_v_j = g[:, None] * (sum_v - v_j)
                    np.add.at(dlat[j], xi[:, j], grad_v_j)
                dlat += L2 * latent[c]
                grads[1][c] += dlat
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * gr * gr
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)
    return linear, latent, bias

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # grid search for k
    best_k, best_acc = 8, -1.0
    for k in K_GRID:
        rng = np.random.default_rng(SEED + k)
        lin, lat, bi = train_fm(X_tr, y_tr, k, VAL_EPOCHS, rng)
        acc = evaluate(X_val, y_val, lin, lat, bi, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # final training on all data
    rng = np.random.default_rng(SEED + best_k * 100)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    linear, latent, bias = train_fm(X, y, best_k, epochs, rng)

    # write weights
    out = struct.pack("<4i", MAGIC, d, best_k, C)
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (argmax of two logits).
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 linear[C][d][256], latent[C][d][256][k], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ( (long)C * d * 256 + (long)C * d * 256 * k + C );
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *latent = linear + (long)C * d * 256;
    const float *bias = latent + (long)C * d * 256 * k;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2], sum_v[32], sum_sq[32];
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        for (int c = 0; c < C; c++) {
            scores[c] = bias[c];
            for (int u = 0; u < k; u++) { sum_v[u] = 0.0f; sum_sq[u] = 0.0f; }
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                scores[c] += linear[((long)c * d + j) * 256 + v];
                const float *lat = latent + (((long)c * d + j) * 256 + v) * k;
                for (int u = 0; u < k; u++) {
                    float val = lat[u];
                    sum_v[u] += val;
                    sum_sq[u] += val * val;
                }
            }
            float inter = 0.0f;
            for (int u = 0; u < k; u++) inter += sum_v[u] * sum_v[u] - sum_sq[u];
            scores[c] += 0.5f * inter;
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```