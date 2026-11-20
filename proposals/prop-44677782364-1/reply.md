```hypothesis
Family: label-onehot-mlp (one-hot encoding + small MLP). Treats each pixel as categorical (0..16) instead of ordinal z-scoring, capturing non-linear intensity effects. A single hidden layer of size H (tuned on last 20% of train) adds feature interactions. Expect >0.9613 dev accuracy (beating label-linear) with cost between label-linear and label-mlp16. If dev accuracy <=0.9613, one-hot encoding alone is insufficient for this task.
```

```python
# train.py for label-onehot-mlp
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Candidate hidden sizes to try on tuning split
H_CANDIDATES = [0, 4, 8, 12, 16]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, d):
    """Build one-hot encoding matrices for each feature from the full training set."""
    enc = []
    for j in range(d):
        vals = np.unique(X[:, j])
        V = len(vals)
        E = np.zeros((256, V), dtype=np.float32)
        E[vals, np.arange(V)] = 1.0
        enc.append(E)
    return enc

def encode_data(X, enc):
    """Encode X using precomputed encoding matrices."""
    d = X.shape[1]
    parts = []
    for j in range(d):
        parts.append(enc[j][X[:, j]])
    return np.concatenate(parts, axis=1)

def train_model(Z, y, H, rng, k, C):
    """Train on encoded data Z (n x k). Returns parameters (W1, b1, W2, b2) and W."""
    n = Z.shape[0]
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
    if H:
        W1, b1, W2, b2 = params
        W = H
    else:
        W1, b1 = params
        W2 = b2 = None
        W = C
    return W1, b1, W2, b2, W

def evaluate(Z, y, W1, b1, W2, b2, H, C):
    """Compute accuracy on encoded data Z."""
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z @ W1 + b1
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == y)

def pack_weights(enc, W1, b1, W2, b2, d, H, C, W):
    """Pack weights into binary format (same as table family)."""
    off = 0
    T_list = []
    for E in enc:
        V = E.shape[1]
        T_feat = E @ W1[off:off + V]
        T_list.append(T_feat.astype(np.float32))
        off += V
    T = np.stack(T_list)  # d x 256 x W
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X.shape[0]
    split = int(0.8 * n_all)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    enc = build_encoding(X, d)
    Z_train = encode_data(X_train, enc)
    Z_tune = encode_data(X_tune, enc)
    k = Z_train.shape[1]
    best_H = 0
    best_acc = -1.0
    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED)
        W1, b1, W2, b2, W = train_model(Z_train, y_train, H, rng, k, C)
        acc = evaluate(Z_tune, y_tune, W1, b1, W2, b2, H, C)
        if acc > best_acc:
            best_acc = acc
            best_H = H
    Z_all = encode_data(X, enc)
    k_all = Z_all.shape[1]
    rng = np.random.default_rng(SEED)
    W1, b1, W2, b2, W = train_model(Z_all, y, best_H, rng, k_all, C)
    weights_bin = pack_weights(enc, W1, b1, W2, b2, d, best_H, C, W)
    open(sys.argv[4], "wb").write(weights_bin)

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