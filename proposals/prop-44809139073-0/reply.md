```hypothesis
Family: label-mlptune (single-hidden-layer MLP with hidden size selected on the tuning split). The Adult dataset mixes ordinal and categorical features; a small MLP can capture nonlinear interactions that logistic regression misses. Tuning H on the last 20% of training data adapts capacity to the task. If dev accuracy does not exceed label-linear (0.8524) or CPU cost exceeds label-mlp16 (2325 Ir/item), the approach fails.
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
H_CHOICES = [4, 8, 16, 32, 64]  # hidden sizes to try


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encodings(X, types):
    """X: (n, d) uint8, types: list of 'ordinal'/'categorical'. Returns list of encoding matrices."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu = v.mean()
            sd = v.std()
            if sd == 0:
                sd = 1.0
            E = ((np.arange(256) - mu) / sd).astype(np.float32)[:, None]  # (256,1)
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encodings(X, enc):
    """X: (n, d) uint8, enc: list of (256, k_j). Returns Z: (n, sum k_j) float32."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_mlp(H, X_tr, y_tr, types, C, seed):
    """Train MLP on given data. Returns (enc, params) where params = (W1, b1, W2, b2) if H>0 else (W1, b1)."""
    enc = compute_encodings(X_tr, types)
    Z = apply_encodings(X_tr, enc)
    n, k = Z.shape
    rng = np.random.default_rng(seed)

    if H > 0:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, dtype=np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, dtype=np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), dtype=np.float32)
        b1 = np.zeros(C, dtype=np.float32)
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y_tr[idx]

            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)

            if H > 0:
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
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return enc, params


def predict(enc, params, H, X, C):
    Z = apply_encodings(X, enc)
    if H > 0:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    return np.argmax(logits, axis=1).astype(np.int64)


def write_weights(enc, params, H, d, C, out_path):
    W = H if H > 0 else C
    # Compute T for each column
    T_list = []
    off = 0
    for E in enc:
        Tj = (E @ params[0][off:off + E.shape[1]]).astype(np.float32)
        T_list.append(Tj)
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)
    # Pack
    magic = 0x54424C31
    header = struct.pack("<4i", magic, d, H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H > 0:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        body += W2 + b2
    with open(out_path, "wb") as f:
        f.write(header + body)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    best_H = None
    best_acc = -1.0
    for H in H_CHOICES:
        enc, params = train_mlp(H, X_tr, y_tr, types, C, seed=SEED)
        preds = predict(enc, params, H, X_tune, C)
        acc = (preds == y_tune).mean()
        print(f"H={H}, tuning accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_H = H

    print(f"Best H={best_H} with accuracy={best_acc:.4f}")
    # Retrain on full data with best_H
    enc_full, params_full = train_mlp(best_H, X_all, y_all, types, C, seed=SEED)
    write_weights(enc_full, params_full, best_H, d, C, sys.argv[4])


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