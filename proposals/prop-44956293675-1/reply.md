```hypothesis
Family: label-mlp8 (single hidden layer of 8 ReLU units).
Why: The optdigits dataset has 64 ordinal pixel features; a tiny nonlinear layer can capture feature interactions that linear models miss, while 8 units keep inference cheaper than label-linear (592 vs 640 multiply-adds per item).
Failure: If dev accuracy ≤ 0.9613 (label-linear) or instructions/item ≥ 5058, the model does not improve on the baseline.
```

```python
"""Table family with a small hidden layer for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Chooses hyperparameters on the last 20% of train.in (file order), then trains on all data.
Exports weights in the same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
H_CHOICES = [4, 8, 12]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
BATCH_CHOICES = [32, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encodings(X, types):
    """Return list of encoding matrices (256 x feat_dim) for each byte position."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Convert byte matrix X (n x d) to feature matrix Z (n x k) using precomputed encodings."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_eval(Z_tr, y_tr, Z_val, y_val, H, lr, l2, batch, rng):
    """Train on Z_tr, y_tr; return validation accuracy and parameters."""
    n, k = Z_tr.shape
    C = int(y_tr.max()) + 1
    W = H if H else C

    # Xavier initialization
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z_tr[idx], y_tr[idx]

            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)

            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + l2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Validation accuracy
    if H:
        a = Z_val @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z_val @ W1 + b1
    pred = scores.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    n = len(X)
    val_size = n // 5
    tr_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    # Hyperparameter search on tuning split
    best_acc = -1.0
    best_params = None
    best_H = None

    for H in H_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                for batch in BATCH_CHOICES:
                    rng = np.random.default_rng(SEED)
                    # Compute encodings on training subset only
                    enc = compute_encodings(X[tr_idx], types)
                    Z_tr = encode(X[tr_idx], enc)
                    Z_val = encode(X[val_idx], enc)
                    acc, params = train_eval(Z_tr, y[tr_idx], Z_val, y[val_idx], H, lr, l2, batch, rng)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (H, lr, l2, batch)
                        best_H = H
    print(f"Best tuning acc: {best_acc:.4f} with H={best_H}, lr={best_params[1]}, l2={best_params[2]}, batch={best_params[3]}", file=sys.stderr)

    # Retrain on full dataset with best hyperparameters
    H, lr, l2, batch = best_params
    rng = np.random.default_rng(SEED)
    enc_full = compute_encodings(X, types)
    Z_full = encode(X, enc_full)
    _, final_params = train_eval(Z_full, y, Z_full, y, H, lr, l2, batch, rng)  # validation not needed

    W1, b1 = final_params[0], final_params[1]
    if H:
        W2, b2 = final_params[2], final_params[3]

    # Build per-byte lookup tables T for inference
    off, T_list = 0, []
    for E in enc_full:
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    W = H if H else C
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T_list).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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