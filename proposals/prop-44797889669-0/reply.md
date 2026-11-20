```hypothesis
Factorization Machine (FM) with rank-8 latent vectors on the label-linear encoded features.
Captures pairwise feature interactions crucial for tabular data (e.g., education × occupation)
while keeping inference fast via folded per-byte lookup tables. Expect ~0.860 accuracy at
~1200 Ir/item (between label-linear and label-mlp16). Fails if accuracy ≤ 0.8524 or cost ≥ 2325 Ir/item.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encodes like label-linear, then learns linear weights + rank-K latent vectors + projection.
Tunes K, LR, L2 on last 20% of train.in (file order), then trains on all data.
Exports per-byte lookup tables: magic 0x464D4C31, d, K, C; float32 T_lin[d][256][C], T_lat[d][256][K],
P[K][C], b_lin[C], b_proj[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Grid searched on tuning split (offline); these are the chosen values.
K = 8
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_EPOCHS = 15
GRID = [
    {"K": 4, "LR": 1e-3, "L2": 1e-4},
    {"K": 8, "LR": 1e-3, "L2": 1e-4},
    {"K": 8, "LR": 3e-3, "L2": 1e-4},
    {"K": 8, "LR": 1e-3, "L2": 1e-3},
]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(card, X):
    """Return list of encoding matrices (256 x d_j) and total encoded dim k."""
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


def build_Z(X, enc):
    """Concatenate encoded features."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def fold_weights(enc, W_lin, V):
    """Fold linear and latent weights into per-byte lookup tables."""
    T_lin, T_lat = [], []
    off = 0
    for E in enc:
        d_j = E.shape[1]
        T_lin.append((E @ W_lin[off:off + d_j]).astype(np.float32))
        T_lat.append((E @ V[off:off + d_j]).astype(np.float32))
        off += d_j
    return np.stack(T_lin), np.stack(T_lat)


def train_fm(Z, y, C, K, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, rng):
    n, k = Z.shape
    # init
    W_lin = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    V = (rng.uniform(-1, 1, (k, K)) * math.sqrt(6.0 / k)).astype(np.float32)
    P = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    b_proj = np.zeros(C, np.float32)
    params = [W_lin, V, P, b_lin, b_proj]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            linear = z @ W_lin + b_lin
            ZV = z @ V
            Z2 = z * z
            V2 = V * V
            ZV2 = Z2 @ V2
            inter = 0.5 * (ZV * ZV - ZV2)
            inter_score = inter @ P + b_proj
            scores = linear + inter_score
            # backward
            dS = softmax_grad(scores, yy)
            # linear
            dW_lin = z.T @ dS + L2 * W_lin
            db_lin = dS.sum(0)
            # projection
            dP = inter.T @ dS + L2 * P
            db_proj = dS.sum(0)
            # interaction
            dinter = dS @ P.T
            dZV = dinter * ZV
            dZV2 = -0.5 * dinter
            # V gradient
            dV = z.T @ dZV + (Z2.T @ dZV2) * (2.0 * V) + L2 * V
            # Adam updates
            grads = [dW_lin, dV, dP, db_lin, db_proj]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W_lin, V, P, b_lin, b_proj


def eval_acc(Z, y, C, W_lin, V, P, b_lin, b_proj):
    linear = Z @ W_lin + b_lin
    ZV = Z @ V
    Z2 = Z * Z
    V2 = V * V
    ZV2 = Z2 @ V2
    inter = 0.5 * (ZV * ZV - ZV2)
    scores = linear + inter @ P + b_proj
    pred = scores.argmax(1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    rng = np.random.default_rng(SEED)
    enc = encode(card, X)
    Z = build_Z(X, enc)
    n, k = Z.shape

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    Z_tr, y_tr = Z[:split], y[:split]
    Z_val, y_val = Z[split:], y[split:]

    # grid search
    best_acc, best_cfg, best_weights = -1, None, None
    for cfg in GRID:
        rng_cfg = np.random.default_rng(SEED + hash(str(cfg)) % 2**32)
        W_lin, V, P, b_lin, b_proj = train_fm(
            Z_tr, y_tr, C, cfg["K"], cfg["LR"], cfg["L2"], BATCH, SAMPLES, TUNE_EPOCHS, rng_cfg
        )
        acc = eval_acc(Z_val, y_val, C, W_lin, V, P, b_lin, b_proj)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
            best_weights = (W_lin.copy(), V.copy(), P.copy(), b_lin.copy(), b_proj.copy())
        print(f"tune K={cfg['K']} LR={cfg['LR']} L2={cfg['L2']} -> val_acc={acc:.4f}", file=sys.stderr)

    # final training on all data
    print(f"best cfg: {best_cfg}, val_acc={best_acc:.4f}", file=sys.stderr)
    W_lin, V, P, b_lin, b_proj = train_fm(
        Z, y, C, best_cfg["K"], best_cfg["LR"], best_cfg["L2"], BATCH, SAMPLES, MAX_EPOCHS, rng
    )

    # fold into per-byte tables
    T_lin, T_lat = fold_weights(enc, W_lin, V)
    # serialize
    magic = 0x464D4C31  # 'FML1'
    K_final = best_cfg["K"]
    out = struct.pack("<4i", magic, d, K_final, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_lat.astype("<f4").tobytes()
    out += P.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += b_proj.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)   // T_lin
        + (size_t)d * 256 * K * sizeof(float)   // T_lat
        + (size_t)K * C * sizeof(float)         // P
        + (size_t)C * sizeof(float)             // b_lin
        + (size_t)C * sizeof(float);            // b_proj
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_lat = T_lin + (size_t)d * 256 * C;
    const float *P = T_lat + (size_t)d * 256 * K;
    const float *b_lin = P + (size_t)K * C;
    const float *b_proj = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_sum = malloc(sizeof(float) * C);
    float *lat_sum = malloc(sizeof(float) * K);
    float *lat_sq_sum = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(lin_sum, b_lin, C * sizeof(float));
        memset(lat_sum, 0, K * sizeof(float));
        memset(lat_sq_sum, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *tlin = T_lin + ((size_t)j * 256 + xi[j]) * C;
            const float *tlat = T_lat + ((size_t)j * 256 + xi[j]) * K;
            for (int c = 0; c < C; c++) lin_sum[c] += tlin[c];
            for (int u = 0; u < K; u++) {
                float v = tlat[u];
                lat_sum[u] += v;
                lat_sq_sum[u] += v * v;
            }
        }
        for (int u = 0; u < K; u++)
            inter[u] = 0.5f * (lat_sum[u] * lat_sum[u] - lat_sq_sum[u]);
        memcpy(scores, b_proj, C * sizeof(float));
        for (int u = 0; u < K; u++) {
            float iu = inter[u];
            const float *pu = P + (size_t)u * C;
            for (int c = 0; c < C; c++) scores[c] += iu * pu[c];
        }
        for (int c = 0; c < C; c++) scores[c] += lin_sum[c];
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```