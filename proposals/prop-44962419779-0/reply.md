```hypothesis
Family: label-mlp8 – a one-hidden-layer MLP with 8 ReLU units trained with Adam.
Why: The Adult dataset benefits from modest non-linear interactions; 8 hidden units add minimal compute (~1k Ir/item) while capturing enough non-linearity to beat the linear baseline (0.8524).
Failure: If dev accuracy ≤ 0.8524 or inference cost exceeds 1500 Ir/item, the model does not improve on label-linear.
```
```python
import json, math, struct, sys, os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
H = 8
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_encodings(X, types):
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std()
            if sd == 0: sd = 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def softmax_grad(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)

def train_loop(Z, y, epochs, H, C, LR, L2, BATCH, rng, init_params=None):
    n, k = Z.shape
    if init_params is None:
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
    else:
        params = init_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ params[0] + params[1]
                h = np.maximum(a, 0)
                g = softmax_grad(h @ params[2] + params[3], yy)
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * params[0],
                    gh.sum(0),
                    h.T @ g + L2 * params[2],
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ params[0] + params[1], yy)
                grads = [
                    z.T @ g + L2 * params[0],
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def evaluate(Z, y, params, H, C):
    if H:
        a = Z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = Z @ params[0] + params[1]
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]

    n = len(X)
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_val = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    # Phase 1: train on 80%, validate on 20% to pick best epoch
    enc_part = compute_encodings(X[idx_train], types)
    Z_train = encode(X[idx_train], enc_part)
    Z_val = encode(X[idx_val], enc_part)
    y_train = y[idx_train]
    y_val = y[idx_val]

    best_epoch = 1
    best_val_acc = -1.0
    params = None
    for epoch in range(1, MAX_EPOCHS + 1):
        params = train_loop(Z_train, y_train, 1, H, C, LR, L2, BATCH, rng, init_params=params)
        val_acc = evaluate(Z_val, y_val, params, H, C)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

    # Phase 2: retrain on full train.in for best_epoch epochs
    enc_full = compute_encodings(X, types)
    Z_full = encode(X, enc_full)
    params = train_loop(Z_full, y, best_epoch, H, C, LR, L2, BATCH, rng)

    # Build lookup tables T (d x 256 x W) and write weights
    W = H if H else C
    off = 0
    T_list = []
    for E in enc_full:
        T_list.append((E @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)

    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()
        out += params[3].astype("<f4").tobytes()

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