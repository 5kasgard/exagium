```hypothesis
Family: label-fm (Factorization Machine). Adult income prediction benefits from pairwise feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A field-aware FM with small latent dimension k=4 adds only ~140KB to weights and ~100 FLOPs/item, capturing interactions efficiently. If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000, the approach fails to improve cost/accuracy tradeoff.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Per-byte lookup tables: linear T[d][256][C], bias[C], latent L[d][256][k][C].
Magic 0x464D3131, d, k, C; float32 T, bias, L.
SGD with Adam, L2 1e-4, minibatch 64, epochs min(100, ceil(600000/n)).
Tunes k in {2,4,8} on last 20% of train, then retrains on all."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(card, X):
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def make_Z(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_fm(Z, y, C, k, rng, n_samples):
    n, F = Z.shape
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    V = (rng.uniform(-1, 1, (F, k, C)) * math.sqrt(6.0 / (F * k))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(n_samples / n))
    Z2 = Z * Z
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, z2, yy = Z[idx], Z2[idx], y[idx]
            B = len(yy)
            # forward
            lin = z @ W + b  # (B, C)
            inter = np.zeros((B, C), np.float32)
            for c in range(C):
                for l in range(k):
                    Vlc = V[:, l, c]
                    sum_l = z @ Vlc
                    sum_sq_l = z2 @ (Vlc * Vlc)
                    inter[:, c] += 0.5 * (sum_l * sum_l - sum_sq_l)
            scores = lin + inter
            # softmax + loss grad
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            dS = (probs / B).astype(np.float32)
            # gradients
            grad_W = z.T @ dS + L2 * W
            grad_b = dS.sum(axis=0)
            grad_V = np.zeros_like(V)
            for c in range(C):
                for l in range(k):
                    Vlc = V[:, l, c]
                    sum_l = z @ Vlc
                    term = dS[:, c:c+1] * (sum_l[:, None] - Vlc[None, :] * z)
                    grad_V[:, l, c] = z.T @ term[:, :, 0] + L2 * Vlc
            grads = [grad_W, grad_V, grad_b]
            # adam
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, V, b


def eval_fm(Z, y, W, V, b, C, k):
    n = Z.shape[0]
    Z2 = Z * Z
    lin = Z @ W + b
    inter = np.zeros((n, C), np.float32)
    for c in range(C):
        for l in range(k):
            Vlc = V[:, l, c]
            sum_l = Z @ Vlc
            sum_sq_l = Z2 @ (Vlc * Vlc)
            inter[:, c] += 0.5 * (sum_l * sum_l - sum_sq_l)
    scores = lin + inter
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def export_weights(card, enc, W, V, b, k, out_path):
    d = card["input"]["bytes"]
    C = card["classes"]
    T_list, L_list = [], []
    off = 0
    for E in enc:
        Fj = E.shape[1]
        W_seg = W[off:off + Fj]
        V_seg = V[off:off + Fj]
        T_list.append(E @ W_seg)
        L_j = np.zeros((256, k, C), np.float32)
        for v in range(256):
            ev = E[v]
            if ev.any():
                for l in range(k):
                    for c in range(C):
                        L_j[v, l, c] = ev @ V_seg[:, l, c]
        L_list.append(L_j)
        off += Fj
    T = np.stack(T_list).astype("<f4")
    L = np.stack(L_list).astype("<f4")
    bias = b.astype("<f4")
    magic = 0x464D3131
    hdr = struct.pack("<4i", magic, d, k, C)
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(T.tobytes())
        f.write(bias.tobytes())
        f.write(L.tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    enc = build_encoders(card, X)
    Z = make_Z(X, enc)
    n = len(y)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]
    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + k)
        W, V, b = train_fm(Z_train, y_train, C, k, rng_k, SAMPLES)
        acc = eval_fm(Z_tune, y_tune, W, V, b, C, k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # retrain on all data with best_k
    rng_best = np.random.default_rng(SEED + best_k + 100)
    W, V, b = train_fm(Z, y, C, best_k, rng_best, SAMPLES)
    export_weights(card, enc, W, V, b, best_k, sys.argv[4])


if __name__ == "__main__":
    main()
```
```c
/* label-fm: Factorization Machine for tabular label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32) { fprintf(stderr, "bad params\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * C;
    const float *L = bias + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        for (int c = 0; c < C; c++) score[c] = bias[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *tj = T + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) score[c] += tj[c];
        }
        for (int l = 0; l < k; l++) { sum_v[l] = 0.0f; sum_sq[l] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lj = L + (((long)j * 256 + v) * k) * C;
            for (int l = 0; l < k; l++) {
                for (int c = 0; c < C; c++) {
                    float val = lj[l * C + c];
                    sum_v[l] += val;
                    sum_sq[l] += val * val;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int l = 0; l < k; l++) inter += 0.5f * (sum_v[l] * sum_v[l] - sum_sq[l]);
            score[c] += inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```