```hypothesis
Family: label-mlp8 (one-hidden-layer MLP with 8 ReLU units, hyperparameters tuned on the last 20% of train).
Why: Optdigits is non-linear; a small MLP adds capacity over linear while per-byte lookup tables keep inference fast (O(d*H) vs O(d*H*C)).
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-mlp16.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid (tuned on validation split)
H_CHOICES = [8, 12, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4]
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_encoders(X, types):
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encoders(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_eval(Z_tr, y_tr, Z_val, y_val, H, C, lr, l2):
    n, k = Z_tr.shape
    rng = np.random.default_rng(SEED)
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
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                logits -= logits.max(axis=1, keepdims=True)
                P = np.exp(logits)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + l2 * W1, gh.sum(0), h.T @ g + l2 * W2, g.sum(0)]
            else:
                logits = z @ W1 + b1
                logits -= logits.max(axis=1, keepdims=True)
                P = np.exp(logits)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                grads = [z.T @ g + l2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # validation accuracy
    if H:
        a_val = Z_val @ W1 + b1
        h_val = np.maximum(a_val, 0)
        logits_val = h_val @ W2 + b2
    else:
        logits_val = Z_val @ W1 + b1
    pred = logits_val.argmax(axis=1)
    return (pred == y_val).mean(), params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    val_size = n // 5
    tr_size = n - val_size
    X_tr, y_tr = X[:tr_size], y[:tr_size]
    X_val, y_val = X[tr_size:], y[tr_size:]

    best_acc = -1.0
    best_cfg = None
    best_enc = None

    for H in H_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                enc = make_encoders(X_tr, card["input"]["types"])
                Z_tr = apply_encoders(X_tr, enc)
                Z_val = apply_encoders(X_val, enc)
                acc, _ = train_eval(Z_tr, y_tr, Z_val, y_val, H, C, lr, l2)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, lr, l2)
                    best_enc = enc

    print(f"Best: H={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    H, lr, l2 = best_cfg
    enc = make_encoders(X, card["input"]["types"])
    Z = apply_encoders(X, enc)
    _, params = train_eval(Z, y, Z, y, H, C, lr, l2)  # validation not used, but function returns params

    if H:
        W1, b1, W2, b2 = params
    else:
        W1, b1 = params
        W2 = b2 = None

    # Export in table format (magic, d, H, C, T, b1, [W2, b2])
    off = 0
    T = []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    W = H if H else C
    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
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