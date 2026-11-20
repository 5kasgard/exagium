```hypothesis
Family: label-mlp-tuned (table family with one ReLU hidden layer, hyperparameters chosen on tuning split).
Why: The linear model (label-linear) ignores interactions between the 16 context positions. A hidden layer can capture pairwise interactions via the summed embeddings. label-mlp16 uses a fixed small hidden size (16) and likely underfits. By tuning hidden size (32-128), learning rate, and L2 on the held-out 20% tuning split, we can find a better capacity/regularization trade-off. The inference cost remains low (same table lookup + small matvec) and should beat label-linear's 0.3375 accuracy.
Failure: If the best tuned model on dev does not exceed 0.3375 accuracy, the added complexity does not help for this task.
```

```python
import json, math, struct, sys

import numpy as np

BASE_SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(X, card):
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode(enc, X):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def init_params(k, W, H, C, rng):
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        return [W1, b1]

def train_model(Z, y, params, H, C, LR, L2, BATCH, EPOCHS, rng):
    n, k = Z.shape
    W = H if H else C
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(EPOCHS):
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
                W1, b1, W2, b2 = params
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                W1, b1 = params
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def evaluate(params, enc, X, y, card, H):
    C = card["classes"]
    Z = encode(enc, X)
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = np.argmax(logits, axis=1)
    return (pred == y).mean()

def write_weights(enc, params, H, C, d, out_path):
    W = H if H else C
    W1 = params[0]
    b1 = params[1]
    off, T = 0, []
    for E in enc:
        T.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = params[2]
        b2 = params[3]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    with open(out_path, "wb") as f:
        f.write(out)

def main():
    if len(sys.argv) != 5:
        print("Usage: train.py card.json train.in train.key weights.bin", file=sys.stderr)
        sys.exit(1)
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    X_all = np.frombuffer(b"".join(records(train_in)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(train_key)], dtype=np.int64)
    n_all = X_all.shape[0]
    tune_size = n_all // 5
    train_size = n_all - tune_size
    X_train, X_tune = X_all[:train_size], X_all[train_size:]
    y_train, y_tune = y_all[:train_size], y_all[train_size:]

    C = card["classes"]
    d = card["input"]["bytes"]

    H_vals = [32, 64, 96, 128]
    LR_vals = [1e-3, 3e-3]
    L2_vals = [1e-4, 1e-3]
    BATCH = 64
    EPOCHS_GRID = 50
    EPOCHS_FULL = 100

    best_acc = -1.0
    best_config = None
    best_enc = None

    for H in H_vals:
        for LR in LR_vals:
            for L2 in L2_vals:
                seed = (H * 1000000 + int(LR * 1e6) + int(L2 * 1e6)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)
                enc = build_encoders(X_train, card)
                Z_train = encode(enc, X_train)
                k = Z_train.shape[1]
                W = H if H else C
                params = init_params(k, W, H, C, rng)
                params = train_model(Z_train, y_train, params, H, C, LR, L2, BATCH, EPOCHS_GRID, rng)
                acc = evaluate(params, enc, X_tune, y_tune, card, H)
                print(f"H={H}, LR={LR}, L2={L2}, tune_acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_config = (H, LR, L2)
                    best_enc = enc

    print(f"Best config: H={best_config[0]}, LR={best_config[1]}, L2={best_config[2]}, acc={best_acc:.4f}")

    H, LR, L2 = best_config
    rng = np.random.default_rng(BASE_SEED)
    enc_full = build_encoders(X_all, card)
    Z_full = encode(enc_full, X_all)
    k = Z_full.shape[1]
    W = H if H else C
    params = init_params(k, W, H, C, rng)
    params = train_model(Z_full, y_all, params, H, C, LR, L2, BATCH, EPOCHS_FULL, rng)
    write_weights(enc_full, params, H, C, d, weights_out)

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