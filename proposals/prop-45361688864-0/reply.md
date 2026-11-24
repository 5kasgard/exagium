```hypothesis
We propose a factorization machine with per-class latent vectors (label-fm). FM captures pairwise feature interactions via factorized parameters, crucial for tabular data like Adult where interactions (e.g., education × occupation) boost accuracy over linear models. With 14 features and latent dimension 8, the model adds only ~2k parameters, keeping inference fast (~2k ops/item). If dev accuracy does not exceed label-linear (0.8524) or latency exceeds label-linear by >2x, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"k": 4, "lr": 0.01, "L2": 1e-4, "epochs": 30},
    {"k": 8, "lr": 0.01, "L2": 1e-4, "epochs": 30},
    {"k": 8, "lr": 0.003, "L2": 1e-4, "epochs": 30},
    {"k": 8, "lr": 0.01, "L2": 1e-5, "epochs": 30},
]
BATCH = 64
SEED = 1
FINAL_EPOCHS = 50
MAGIC = 0x464D4C31  # 'FML1' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of (E, dim, offset) and total_dim."""
    encoders = []
    offset = 0
    for j, t in enumerate(types):
        col = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            E = ((np.arange(256) - mu) / sd).astype(np.float32).reshape(256, 1)
            dim = 1
        else:
            vals = np.unique(col)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            dim = len(vals)
        encoders.append((E, dim, offset))
        offset += dim
    return encoders, offset


def encode(X, encoders, total_dim):
    n = X.shape[0]
    Z = np.zeros((n, total_dim), dtype=np.float32)
    for j, (E, dim, offset) in enumerate(encoders):
        Z[:, offset:offset + dim] = E[X[:, j]]
    return Z


def train_fm(Z, y, C, total_dim, k, lr, L2, batch, epochs, seed):
    rng = np.random.default_rng(seed)
    # Xavier initialization
    W = (rng.uniform(-1, 1, (total_dim, C)) * np.sqrt(6.0 / total_dim)).astype(np.float32)
    V = (rng.uniform(-1, 1, (total_dim, C, k)) * np.sqrt(6.0 / (total_dim * k))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = Z.shape[0]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Zb = Z[idx]
            yb = y[idx]
            B = len(yb)

            # Forward
            linear = Zb @ W  # (B, C)
            latent = np.zeros((B, C, k), dtype=np.float32)
            latent_sq = np.zeros((B, C, k), dtype=np.float32)
            for c in range(C):
                Vc = V[:, c, :]
                latent[:, c, :] = Zb @ Vc
                latent_sq[:, c, :] = (Zb ** 2) @ (Vc ** 2)
            interaction = 0.5 * np.sum(latent ** 2 - latent_sq, axis=2)  # (B, C)
            scores = linear + interaction

            # Softmax cross-entropy gradient
            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Gradients
            grad_W = Zb.T @ grad_scores + L2 * W
            grad_V = np.zeros_like(V)
            for c in range(C):
                Vc = V[:, c, :]
                sum_v = latent[:, c, :]
                g = grad_scores[:, c]
                term1 = Zb.T @ (g[:, None] * sum_v)
                term2 = (Zb ** 2).T @ (g[:, None] * Vc)
                grad_V[:, c, :] = term1 - term2 + L2 * Vc
            grad_b = grad_scores.sum(axis=0)

            # Adam update
            t += 1
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            W -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            b -= lr * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    return W, V, b


def predict(Z, W, V, b, C):
    n = Z.shape[0]
    linear = Z @ W
    latent = np.zeros((n, C, V.shape[2]), dtype=np.float32)
    latent_sq = np.zeros((n, C, V.shape[2]), dtype=np.float32)
    for c in range(C):
        Vc = V[:, c, :]
        latent[:, c, :] = Z @ Vc
        latent_sq[:, c, :] = (Z ** 2) @ (Vc ** 2)
    interaction = 0.5 * np.sum(latent ** 2 - latent_sq, axis=2)
    scores = linear + interaction
    return np.argmax(scores, axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]

    encoders, total_dim = build_encoders(X, types)
    Z = encode(X, encoders, total_dim)
    n = len(X)

    # Tuning split: last 20% in file order
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_val, y_val = Z[val_idx], y[val_idx]

    best_acc = -1.0
    best_cfg = None
    best_weights = None

    for cfg in GRID:
        W, V, b = train_fm(Z_train, y_train, C, total_dim, cfg["k"], cfg["lr"],
                           cfg["L2"], BATCH, cfg["epochs"], SEED)
        preds = predict(Z_val, W, V, b, C)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_weights = (W, V, b)

    # Retrain on full data with best config
    W, V, b = train_fm(Z, y, C, total_dim, best_cfg["k"], best_cfg["lr"],
                       best_cfg["L2"], BATCH, FINAL_EPOCHS, SEED)

    # Build per-byte lookup tables
    k = best_cfg["k"]
    T_lin = np.zeros((d, 256, C), dtype=np.float32)
    T_lat = np.zeros((d, 256, C, k), dtype=np.float32)
    for j, (E, dim, offset) in enumerate(encoders):
        W_feat = W[offset:offset + dim]          # (dim, C)
        V_feat = V[offset:offset + dim]          # (dim, C, k)
        T_lin[j] = E @ W_feat
        for c in range(C):
            T_lat[j, :, c, :] = E @ V_feat[:, c, :]

    # Write weights.bin
    header = struct.pack("<4i", MAGIC, d, C, k)
    body = T_lin.astype("<f4").tobytes() + T_lat.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


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
    int d = hd[1], C = hd[2], k = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * C * k + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_lat = T_lin + (long)d * 256 * C;
    const float *b = T_lat + (long)d * 256 * C * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            for (int u = 0; u < k; u++) { sum_v[u] = 0.0f; sum_sq[u] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *lin = T_lin + ((long)j * 256 + xi[j]) * C;
                scores[c] += lin[c];
                const float *lat = T_lat + (((long)j * 256 + xi[j]) * C + c) * k;
                for (int u = 0; u < k; u++) {
                    float v = lat[u];
                    sum_v[u] += v;
                    sum_sq[u] += v * v;
                }
            }
            float inter = 0.0f;
            for (int u = 0; u < k; u++) inter += sum_v[u] * sum_v[u] - sum_sq[u];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```