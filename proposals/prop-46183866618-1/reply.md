```hypothesis
We propose "label-mlp32": a one-hidden-layer MLP with 32 ReLU units, using per-byte lookup tables for the first layer (like label-linear) to keep inference fast. The 64 ordinal pixel features (0..16) in optdigits have non-linear interactions that a linear model cannot capture; a small MLP is known to excel here (label-mlp16 gets 0.977). We expect dev accuracy > 0.97 (beating label-linear's 0.9613) with instruction count around 10k Ir/item (still far below label-knn). If dev accuracy ≤ 0.965 or instruction count exceeds 20k Ir/item, the model does not meet its goals.
```

```python
"""label-mlp32: one-hidden-layer MLP with per-byte lookup tables, hyperparameter chosen on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (same as label-linear)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_H = [0, 8, 16, 32, 64]  # hidden sizes to try on validation split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit_indices=None):
    """Return list of encoding matrices (256 x dim_j) and the concatenated Z matrix for indices fit_indices."""
    n, d = X.shape
    if fit_indices is None:
        fit_indices = np.arange(n)
    enc = []
    for j, t in enumerate(types):
        v = X[fit_indices, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return enc, Z


def train_model(Z, y, H, C, rng, n_epochs):
    """Train on given data, return params [W1, b1, (W2, b2)]."""
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
    for _ in range(n_epochs):
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
    """Return accuracy on given data."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def build_tables(enc, W1, d, H):
    """Build per-byte lookup tables T[d][256][W] (W = H if H>0 else C)."""
    W = H if H else W1.shape[1]
    T = np.zeros((d, 256, W), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = E.shape[1]
        T[j] = (E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32)
        off += dim
    return T


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = len(X)

    # Split: last 20% for validation (tuning split)
    val_start = int(0.8 * n)
    train_idx = np.arange(val_start)
    val_idx = np.arange(val_start, n)

    best_H = 0
    best_val_acc = -1.0
    best_params = None
    best_enc = None

    for H in CANDIDATE_H:
        rng = np.random.default_rng(SEED)
        # Encode using training split statistics
        enc, Z_full = encode_features(X, types, fit_indices=train_idx)
        Z_train, y_train = Z_full[train_idx], y[train_idx]
        Z_val, y_val = Z_full[val_idx], y[val_idx]

        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
        params = train_model(Z_train, y_train, H, C, rng, n_epochs)
        val_acc = evaluate(Z_val, y_val, params, H, C)
        print(f"H={H}, val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_H = H
            best_params = params
            best_enc = enc

    print(f"Selected H={best_H} with val_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best_H
    rng = np.random.default_rng(SEED)
    enc_full, Z_full = encode_features(X, types)  # uses all data for stats
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(Z_full, y, best_H, C, rng, n_epochs)

    # Build tables for inference
    W1 = final_params[0]
    b1 = final_params[1]
    T = build_tables(enc_full, W1, d, best_H)
    W = best_H if best_H else C

    # Pack weights: magic, d, H, C; then T (d*256*W floats), b1 (W floats); if H>0: W2, b2
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
        W2, b2 = final_params[2], final_params[3]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp32 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weight format: magic 0x54424C31, int32 d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H if H>0 else C). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k>>8) & 255, (k>>16) & 255, (k>>24) & 255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hdr = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hdr[0] != 0x54424C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], H = hdr[2], C = hdr[3];
    int W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hdr + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *acc = (float *)malloc(sizeof(float) * W);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld: length %u != %d\n", i, k, d); return 3; }

        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }

        const float *logits = acc;
        if (H) {
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float hu = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) scores[c] += hu * W2[(long)u * C + c];
            }
            logits = scores;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```