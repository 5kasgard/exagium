```hypothesis
Family: label-shallow-mlp (one-hidden-layer ReLU MLP with per-position byte embeddings, hyperparameters tuned on the last 20% of train.in).
Why: The task is next-character prediction from 16 bytes of context. A linear model (label-linear, 0.3375) cannot capture interactions between positions; a hidden layer can. The existing label-mlp16 (H=16) underperforms (0.3260), likely because its hyperparameters (epochs, LR, L2) were not tuned for a hidden layer. We tune H ∈ {64,128}, LR ∈ {3e-3,1e-2}, L2 ∈ {1e-4}, epochs ∈ {25,50} on the tuning split, then retrain on all data. This should beat linear while keeping inference cost ~5× linear (still < 50k Ir/item).
Failure signal: If best tuned configuration on dev does not exceed 0.34 accuracy, the family does not justify its extra cost.
```

```python
"""label-shallow-mlp: one-hidden-layer ReLU MLP with per-position byte embeddings, hyperparameter tuning on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1 for final training), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
H_CHOICES = [64, 128]
LR_CHOICES = [3e-3, 1e-2]
L2_CHOICES = [1e-4]
SAMPLES_CHOICES = [600000, 1200000]   # 25 or 50 epochs for n=24000
BATCH = 64
MAX_EPOCHS = 100
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256, dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode_batch(X, enc):
    """X: (n, d) uint8 -> Z: (n, k) float32"""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_mlp(Z, y, C, H, LR, L2, SAMPLES, rng):
    """Train on given data, return params (W1,b1,W2,b2) and T list."""
    n, k = Z.shape
    W = H if H else C
    # Initialize
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
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
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
    # Build per-position tables T
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    return params, T

def evaluate(Z, y, params, H, C):
    n = Z.shape[0]
    W1, b1 = params[0], params[1]
    if H:
        W2, b2 = params[2], params[3]
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()

def pack_weights(d, H, C, T, params):
    W = H if H else C
    magic = 0x54424C32  # "TBL2"
    out = struct.pack("<4i", magic, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X.shape[0]
    # Tuning split: last 20% in file order
    n_tune = n_total // 5
    n_train = n_total - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    # Build encoding on train split only
    global enc
    enc = build_encoding(X_train, types)
    Z_train = encode_batch(X_train, enc)
    Z_tune = encode_batch(X_tune, enc)

    best_acc = -1.0
    best_cfg = None
    best_params = None
    best_T = None

    for H in H_CHOICES:
        for LR in LR_CHOICES:
            for L2 in L2_CHOICES:
                for SAMPLES in SAMPLES_CHOICES:
                    # Deterministic seed per configuration
                    cfg_seed = SEED + H + int(LR*10000) + int(L2*10000) + SAMPLES//1000
                    rng = np.random.default_rng(cfg_seed)
                    params, T = train_mlp(Z_train, y_train, C, H, LR, L2, SAMPLES, rng)
                    acc = evaluate(Z_tune, y_tune, params, H, C)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (H, LR, L2, SAMPLES)
                        best_params = params
                        best_T = T
    # Retrain on full data with best config
    H, LR, L2, SAMPLES = best_cfg
    enc = build_encoding(X, types)
    Z_full = encode_batch(X, enc)
    rng = np.random.default_rng(SEED)
    params, T = train_mlp(Z_full, y, C, H, LR, L2, SAMPLES, rng)
    # Save
    out = pack_weights(d, H, C, T, params)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-shallow-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weight format: magic 0x54424C32, d, H, C; float32 T[d][256][W], b1[W]; if H: W2[H][C], b2[C] (W=H if H else C). */
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
    if (nw < 16 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *s = malloc(sizeof(float) * C);
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