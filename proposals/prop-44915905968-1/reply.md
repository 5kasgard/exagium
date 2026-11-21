```hypothesis
We propose "label-mlp-tuned", a one-hidden-layer ReLU MLP that selects hidden size H ∈ {4,8,12,16} on the tuning split (last 20% of train.in). Optdigits pixels exhibit local spatial correlations; a single non-linear layer can capture simple stroke patterns that a linear model misses. With H=8 the inference cost (~600 multiply-adds) is comparable to label-linear, while accuracy should approach label-mlp16 (97.7%). If dev accuracy ≤ label-linear (96.1%) or cost exceeds label-mlp16, the hypothesis fails.
```
```python
"""label-mlp-tuned: one-hidden-layer ReLU MLP with hidden size chosen on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weight format identical to label-linear (table family) so the same infer.c works."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (except H which we tune)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_CHOICES = [4, 8, 12, 16]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256, dim_j) and feature means/stds for ordinal."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd).astype(np.float32)[:, None]  # (256,1)
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode_data(X, enc):
    """X (n,d) uint8 -> Z (n, total_dim) float32 using precomputed enc."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def init_params(k, H, C, rng):
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - one_hot) / batch."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    P = exp / exp.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)


def forward(params, z, H, C):
    if H:
        a = z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = z @ params[0] + params[1]
    return logits


def accuracy(params, Z, y, H, C):
    logits = forward(params, Z, H, C)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(Z_train, y_train, Z_val, y_val, H, C, rng_seed):
    n, k = Z_train.shape
    rng = np.random.default_rng(rng_seed)
    params = init_params(k, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            if H:
                a = z @ params[0] + params[1]
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]
                g = softmax_grad(logits, yy)
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * params[0],
                    gh.sum(0),
                    h.T @ g + L2 * params[2],
                    g.sum(0)
                ]
            else:
                logits = z @ params[0] + params[1]
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + L2 * params[0],
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_acc = accuracy(params, Z_val, y_val, H, C)
    return params, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Split: last 20% for tuning (validation)
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Build encoding on training split
    enc_train = build_encoding(X_train, types)
    Z_train = encode_data(X_train, enc_train)
    Z_val = encode_data(X_val, enc_train)

    # Grid search over H
    best_H, best_acc, best_params = None, -1, None
    for H in H_CHOICES:
        params, acc = train_model(Z_train, y_train, Z_val, y_val, H, C, SEED + H * 1000)
        print(f"H={H} val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_H, best_params = acc, H, params

    print(f"Selected H={best_H} with val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset with best_H
    enc_full = build_encoding(X, types)
    Z_full = encode_data(X, enc_full)
    final_params, _ = train_model(Z_full, y, Z_full, y, best_H, C, SEED + best_H * 1000 + 1)

    # Pack weights into the table-family format (same as label-linear)
    d = X.shape[1]
    W = best_H if best_H else C
    # Compute T: for each feature j, T_j = E_j @ W1_slice
    # final_params[0] is W1 (k, W) where k = d (since each ordinal gives 1 dim)
    T = []
    off = 0
    for E in enc_full:
        # E is (256,1), W1 slice is (1, W)
        T.append((E @ final_params[0][off:off+1]).astype(np.float32))
        off += 1
    T = np.stack(T)  # (d, 256, W)

    # Write weights.bin
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += final_params[1].astype("<f4").tobytes()  # b1
    if best_H:
        out += final_params[2].astype("<f4").tobytes()  # W2
        out += final_params[3].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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