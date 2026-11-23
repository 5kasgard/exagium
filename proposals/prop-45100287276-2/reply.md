```hypothesis
We propose a two-layer MLP (label-mlp-tuned) with ReLU activations on one-hot encoded byte positions.
Hyperparameters (hidden size, learning rate, L2) are selected on the last 20% tuning split, then the model is retrained on all data.
This captures pairwise interactions between context bytes that a linear model misses.
If dev accuracy does not exceed label-linear's 0.3375, the family offers no advantage.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning
H_CHOICES = [0, 64, 128, 256]   # 0 = linear
LR_CHOICES = [3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
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

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def train_model(X_tr, y_tr, X_val, y_val, enc, H, LR, L2, rng_seed):
    """Train on (X_tr, y_tr), evaluate on (X_val, y_val). Return (val_acc, params)."""
    n_tr, d = X_tr.shape
    C = 65  # from card
    # Build design matrices
    Z_tr = np.concatenate([E[X_tr[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    Z_val = np.concatenate([E[X_val[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    k = Z_tr.shape[1]
    W = H if H else C
    rng = np.random.default_rng(rng_seed)
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
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            if H:
                a = z @ params[0] + params[1]
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]
            else:
                logits = z @ params[0] + params[1]
            # softmax gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            if H:
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * params[0],
                    gh.sum(0),
                    h.T @ g + L2 * params[2],
                    g.sum(0)
                ]
            else:
                grads = [
                    z.T @ g + L2 * params[0],
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    if H:
        a = Z_val @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = Z_val @ params[0] + params[1]
    preds = logits.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params

def write_weights(card, enc, params, H, out_path):
    d = card["input"]["bytes"]
    C = card["classes"]
    W = H if H else C
    # T is list of d matrices (256 x W)
    T = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ params[0][off:off+dim]).astype(np.float32))
        off += dim
    T = np.stack(T)  # d x 256 x W
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()
        out += params[3].astype("<f4").tobytes()
    open(out_path, "wb").write(out)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    enc = build_encoding(X_tr, card["input"]["types"])
    best_acc = -1
    best_cfg = None
    best_params = None
    for H in H_CHOICES:
        for LR in LR_CHOICES:
            for L2 in L2_CHOICES:
                acc, params = train_model(X_tr, y_tr, X_val, y_val, enc, H, LR, L2, SEED)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (H, LR, L2)
                    best_params = params
    print(f"Best config: H={best_cfg[0]}, LR={best_cfg[1]}, L2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    H, LR, L2 = best_cfg
    enc_full = build_encoding(X, card["input"]["types"])
    _, final_params = train_model(X, y, X[:1], y[:1], enc_full, H, LR, L2, SEED)  # dummy val
    write_weights(card, enc_full, final_params, H, sys.argv[4])

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