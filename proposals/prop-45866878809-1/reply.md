```hypothesis
Family: label-mlp-tuned – a one-hidden-layer MLP (ReLU, Adam) with hidden size H chosen on the tuning split (last 20% of train). The lookup-table inference keeps latency near label-linear while non-linearity should lift accuracy on optdigits' pixel interactions. Failure if tuning selects H=0 (linear) or if the small dataset causes overfitting so dev accuracy ≤ label-linear.
```

```python
"""label-mlp-tuned: one-hidden-layer MLP with ReLU, Adam, and per-feature lookup tables.
Hyperparameter H (hidden units) is chosen on the tuning split (last 20% of train.in),
then the model is retrained on all training data. Weights format matches the table family.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (same as label-linear example)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# Candidate hidden sizes to try on the tuning split
H_CANDIDATES = [0, 4, 8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encoding(X, types):
    """Return list of encoding matrices E_j (256 x W_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])   # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoding(X, enc):
    """Encode X (n x d) using list of encodings -> Z (n x sum W_j)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_mlp(Z, y, H, C, rng, n_epochs):
    """Train MLP on encoded data Z (n x k). Returns list of parameters."""
    n, k = Z.shape
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate_accuracy(X, y, enc, params, H, C):
    Z = apply_encoding(X, enc)
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        W1, b1 = params
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def build_tables(enc, params, H, C):
    """Construct per-feature lookup tables T (d x 256 x W) and biases."""
    W = H if H else C
    d = len(enc)
    T = []
    off = 0
    W1 = params[0]
    for E in enc:
        w = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + w]).astype(np.float32))  # (256, W)
        off += w
    b1 = params[1].astype(np.float32)
    return T, b1


def write_weights(weights_path, d, H, C, T, b1, params):
    """Write weights.bin in the table family format."""
    W = H if H else C
    magic = 0x54424C31
    header = struct.pack("<4i", magic, d, H, C)
    body = np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        body += W2 + b2
    open(weights_path, "wb").write(header + body)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]

    n = len(X_all)
    tune_size = int(0.2 * n)
    train_size = n - tune_size
    X_train, X_tune = X_all[:train_size], X_all[train_size:]
    y_train, y_tune = y_all[:train_size], y_all[train_size:]

    best_H, best_acc = -1, -1.0
    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED)
        enc = compute_encoding(X_train, types)
        Z_train = apply_encoding(X_train, enc)
        Z_tune = apply_encoding(X_tune, enc)

        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_mlp(Z_train, y_train, H, C, rng, n_epochs)
        acc = evaluate_accuracy(X_tune, y_tune, enc, params, H, C)
        print(f"H={H} tune_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_H = acc, H

    print(f"Selected H={best_H} with tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best_H
    rng = np.random.default_rng(SEED)
    enc_full = compute_encoding(X_all, types)
    Z_all = apply_encoding(X_all, enc_full)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params_final = train_mlp(Z_all, y_all, best_H, C, rng, n_epochs)
    T, b1 = build_tables(enc_full, params_final, best_H, C)
    write_weights(sys.argv[4], d, best_H, C, T, b1, params_final)


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