```hypothesis
Family: label-fm (factorization machine with linear term). The Adult dataset has strong pairwise interactions between categorical features (e.g., education × occupation). A factorization machine captures these with low-rank embeddings (size K) while retaining a per-feature linear term. Inference adds only O(d·K + K·C) operations over the linear baseline, staying fast. If dev accuracy does not exceed label-linear (0.8524) or CPU cost grows significantly, the approach fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on the tuning split (last 20% of train.in)
GRID = [
    {"K": 4, "lr": 0.01, "l2": 1e-4},
    {"K": 8, "lr": 0.01, "l2": 1e-4},
    {"K": 8, "lr": 0.003, "l2": 1e-4},
    {"K": 16, "lr": 0.003, "l2": 1e-4},
]
SEED = 1
BATCH = 64
MAX_EPOCHS_GRID = 10
MAX_EPOCHS_FULL = 50
SAMPLES_PER_EPOCH = 600000


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(card, X):
    """Return list of encoding matrices E_j (256, dim_j) and total dim."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def init_params(d, C, K, rng):
    # Linear weights: (d, 256, C)
    L = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / 256)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    # Embeddings: (d, 256, K)
    E = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Interaction projection: (K, C)
    W2 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b_int = np.zeros(C, np.float32)
    return L, b_lin, E, W2, b_int


def forward(X_idx, L, b_lin, E, W2, b_int):
    """X_idx: (n, d) uint8. Returns scores (n, C)."""
    n, d = X_idx.shape
    # Linear part
    scores = b_lin + np.sum(L[np.arange(d), X_idx.T].transpose(1, 0, 2), axis=1)  # (n, C)
    # Embedding sum
    emb_sum = np.sum(E[np.arange(d), X_idx.T].transpose(1, 0, 2), axis=1)  # (n, K)
    # Interaction part
    scores += b_int + emb_sum @ W2
    return scores


def softmax_cross_entropy_grad(scores, y):
    """Return gradient of loss w.r.t. scores (n, C)."""
    n = scores.shape[0]
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # (n, C)


def train_one_config(X_idx, y, d, C, K, lr, l2, epochs, rng):
    L, b_lin, E, W2, b_int = init_params(d, C, K, rng)
    params = [L, b_lin, E, W2, b_int]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    n = X_idx.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_idx[idx]
            yb = y[idx]
            # Forward
            # Linear
            lin_part = b_lin + np.sum(L[np.arange(d), xb.T].transpose(1, 0, 2), axis=1)
            # Embedding sum
            emb_sum = np.sum(E[np.arange(d), xb.T].transpose(1, 0, 2), axis=1)
            # Interaction
            scores = lin_part + b_int + emb_sum @ W2
            # Loss gradient
            g_scores = softmax_cross_entropy_grad(scores, yb)  # (b, C)
            # Gradients
            # Linear weights and bias
            g_L = np.zeros_like(L)
            g_b_lin = g_scores.sum(axis=0)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    g_L[j, vals, c] += g_scores[:, c]
            g_L += l2 * L
            g_b_lin += l2 * b_lin
            # Embeddings and W2, b_int
            g_emb_sum = g_scores @ W2.T  # (b, K)
            g_W2 = emb_sum.T @ g_scores + l2 * W2  # (K, C)
            g_b_int = g_scores.sum(axis=0) + l2 * b_int
            g_E = np.zeros_like(E)
            for j in range(d):
                vals = xb[:, j]
                for k in range(K):
                    g_E[j, vals, k] += g_emb_sum[:, k]
            g_E += l2 * E
            grads = [g_L, g_b_lin, g_E, g_W2, g_b_int]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X_idx, y, params):
    L, b_lin, E, W2, b_int = params
    scores = forward(X_idx, L, b_lin, E, W2, b_int)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_cfg = None

    for cfg in GRID:
        cfg_rng = np.random.default_rng(SEED + hash(str(cfg)) % 2**32)
        params = train_one_config(X_train, y_train, d, C, cfg["K"], cfg["lr"], cfg["l2"], MAX_EPOCHS_GRID, cfg_rng)
        acc = evaluate(X_tune, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg

    # Retrain on full data with best config
    full_rng = np.random.default_rng(SEED + 12345)
    final_params = train_one_config(X, y, d, C, best_cfg["K"], best_cfg["lr"], best_cfg["l2"], MAX_EPOCHS_FULL, full_rng)

    L, b_lin, E, W2, b_int = final_params
    # Pack weights: magic, d, K, C, then arrays in order
    magic = 0x464D4C31  # "FML1"
    K = best_cfg["K"]
    header = struct.pack("<4i", magic, d, K, C)
    # Linear weights: d * 256 * C
    lin_bytes = L.astype("<f4").tobytes()
    # Linear bias: C
    lin_bias_bytes = b_lin.astype("<f4").tobytes()
    # Embeddings: d * 256 * K
    emb_bytes = E.astype("<f4").tobytes()
    # W2: K * C
    w2_bytes = W2.astype("<f4").tobytes()
    # b_int: C
    b_int_bytes = b_int.astype("<f4").tobytes()
    out = header + lin_bytes + lin_bias_bytes + emb_bytes + w2_bytes + b_int_bytes
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 16
        + 4L * d * 256 * C          // linear weights
        + 4L * C                    // linear bias
        + 4L * d * 256 * K          // embeddings
        + 4L * K * C                // W2
        + 4L * C;                   // b_int
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *L = p; p += (long)d * 256 * C;
    const float *b_lin = p; p += C;
    const float *E = p; p += (long)d * 256 * K;
    const float *W2 = p; p += (long)K * C;
    const float *b_int = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = calloc(K, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int j = 0; j < d; j++) {
            const float *lj = L + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        // Embedding sum
        for (int kk = 0; kk < K; kk++) emb_sum[kk] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) emb_sum[kk] += ej[kk];
        }

        // Interaction part
        for (int c = 0; c < C; c++) {
            float acc = b_int[c];
            for (int kk = 0; kk < K; kk++) acc += emb_sum[kk] * W2[(long)kk * C + c];
            scores[c] += acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```