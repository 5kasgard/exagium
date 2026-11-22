```hypothesis
Family: label-mlp-tuned. A one-hidden-layer ReLU MLP with per-byte lookup tables (same architecture as label-linear/label-mlp16), but hidden size H is tuned on the validation split (last 20% of train.in) from {4,8,12,16}. This adds non-linearity to beat label-linear (H=0) while potentially using a smaller H than label-mlp16 (H=16) for lower inference cost. If no H>0 improves validation accuracy over linear, the family fails.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_CANDIDATES = [4, 8, 12, 16]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def prepare_data(X, y, card, encoders=None):
    """Compute or apply per-feature encoders. Returns Z (n, k) and encoders list."""
    n, d = X.shape
    if encoders is None:
        encoders = []
        for j, t in enumerate(card["input"]["types"]):
            v = X[:, j].astype(np.float64)
            if t == "ordinal":
                mu, sd = v.mean(), v.std() or 1.0
                encoders.append(("ordinal", mu, sd))
            else:
                vals = np.unique(X[:, j])
                E = np.zeros((256, len(vals)), np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                encoders.append(("categorical", E))
    # Build Z
    Z_parts = []
    for j, enc in enumerate(encoders):
        if enc[0] == "ordinal":
            mu, sd = enc[1], enc[2]
            col = ((X[:, j].astype(np.float32) - mu) / sd).reshape(-1, 1)
            Z_parts.append(col)
        else:
            E = enc[1]
            Z_parts.append(E[X[:, j]])
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, encoders

def train_model(Z, y, H, C, rng):
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
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
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
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def evaluate(Z, y, params, H, C):
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

def pack_weights(card, encoders, params, H, C, d):
    """Pack weights into the same format as the example."""
    W = H if H else C
    # Reconstruct the example's enc matrices
    enc_matrices = []
    for enc in encoders:
        if enc[0] == "ordinal":
            mu, sd = enc[1], enc[2]
            E = ((np.arange(256) - mu) / sd).reshape(-1, 1).astype(np.float32)
            enc_matrices.append(E)
        else:
            enc_matrices.append(enc[1].astype(np.float32))
    off = 0
    T_all = []
    for E in enc_matrices:
        T_all.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_all).astype("<f4")
    b1 = params[1].astype("<f4")
    out = struct.pack("<4i", 0x54424C31, d, H, C) + T.tobytes() + b1.tobytes()
    if H:
        W2 = params[2].astype("<f4")
        b2 = params[3].astype("<f4")
        out += W2.tobytes() + b2.tobytes()
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # Validation split: last 20% in file order
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    best_H = None
    best_acc = -1.0
    best_params = None
    best_encoders = None

    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED)
        Z_train, encoders = prepare_data(X_train, y_train, card)
        params = train_model(Z_train, y_train, H, C, rng)
        Z_val, _ = prepare_data(X_val, y_val, card, encoders)
        acc = evaluate(Z_val, y_val, params, H, C)
        print(f"H={H}, val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_H = H
            best_params = params
            best_encoders = encoders

    print(f"Best H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best_H
    rng = np.random.default_rng(SEED)
    Z_full, encoders_full = prepare_data(X, y, card)
    final_params = train_model(Z_full, y, best_H, C, rng)
    weights = pack_weights(card, encoders_full, final_params, best_H, C, d)
    open(sys.argv[4], "wb").write(weights)

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