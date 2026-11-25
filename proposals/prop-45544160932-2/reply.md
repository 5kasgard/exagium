```hypothesis
Family: label-mlp (single hidden-layer MLP with tuned hidden size, ReLU, Adam).
Why: The 16-byte context has strong non-linear interactions (e.g., letter pairs, word boundaries) that a linear model cannot capture. A modest hidden layer (32–128 units) adds negligible inference cost (still one matrix-vector product per layer) while modeling these interactions. Tuning H, LR, L2 on the held-out 20% avoids overfitting the small 24k training set.
Failure signal: dev accuracy ≤ 0.34 (not beating label-linear) or instruction count > 2× label-linear.
```

```python
"""label-mlp: one-hidden-layer MLP with per-byte one-hot embeddings, ReLU, Adam.
Hyperparameters (H, LR, L2) chosen on last 20% of train.in, then retrained on all train.
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Weights format identical to label-linear (magic 0x54424C31, d, H, C; T[d][256][W], b1[W]; if H: W2[H][C], b2[C])."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS = 50          # for both tuning and final training
SAMPLES = 600000         # max samples per epoch (like label-linear)
# Hyperparameter grid
H_GRID = [32, 64, 128]
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def make_encodings(X, types):
    """Return list of (256, nvals) float64 one-hot matrices for each column."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode(X, enc):
    """Concatenate one-hot columns -> (n, total_features) float32."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_loop(Z, y, H, C, LR, L2, rng, epochs):
    """Train on (Z,y) for given epochs, return params [W1,b1,W2?,b2?]."""
    n, k = Z.shape
    W = H if H else C
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
                grads = [z.T @ gh + L2 * W1, gh.sum(0),
                         h.T @ g + L2 * W2, g.sum(0)]
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

def eval_acc(Z, y, params, H, C):
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        S = h @ W2 + b2
    else:
        W1, b1 = params
        S = Z @ W1 + b1
    pred = S.argmax(axis=1)
    return (pred == y).mean()

def pack_weights(card, enc, params, H, C):
    """Serialize to the label-linear binary format."""
    d = card["input"]["bytes"]
    W = H if H else C
    # T: per-byte lookup tables (d, 256, W)
    T_list = []
    off = 0
    for E in enc:
        # E is (256, nvals), params[0] is (total_features, W)
        T_list.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.tobytes()
    if H:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        out += W2 + b2
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]

    # Build encodings on full training data (for final) and on train split (for tuning)
    enc_full = make_encodings(X, card["input"]["types"])
    Z_full = encode(X, enc_full)

    # For tuning we need encodings based only on train split
    enc_tune = make_encodings(X[train_idx], card["input"]["types"])
    Z_train = encode(X[train_idx], enc_tune)
    Z_tune = encode(X[tune_idx], enc_tune)
    y_train, y_tune = y[train_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None

    for H in H_GRID:
        for LR in LR_GRID:
            for L2 in L2_GRID:
                cfg_rng = np.random.default_rng(SEED)  # same init for each config
                params = train_loop(Z_train, y_train, H, C, LR, L2, cfg_rng, MAX_EPOCHS)
                acc = eval_acc(Z_tune, y_tune, params, H, C)
                if acc > best_acc:
                    best_acc, best_cfg, best_params = acc, (H, LR, L2), params
                    print(f"Tune: H={H} LR={LR} L2={L2} acc={acc:.4f}", file=sys.stderr)

    H, LR, L2 = best_cfg
    print(f"Best: H={H} LR={LR} L2={L2} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    final_rng = np.random.default_rng(SEED)
    final_params = train_loop(Z_full, y, H, C, LR, L2, final_rng, MAX_EPOCHS)
    weights = pack_weights(card, enc_full, final_params, H, C)
    open(sys.argv[4], "wb").write(weights)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: single hidden-layer MLP with per-byte lookup tables.
 * Format: magic 0x54424C31, int32 d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H). */
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