```hypothesis
Family: label-factorized (factorized linear model with per-position byte embeddings summed, then a ReLU hidden layer, then linear to classes). Why: The linear model's 4096×65 weight matrix is approximated by 16×256×D embeddings + D×65 output weights (D=32..64). The bottleneck forces sharing across positions and the ReLU adds non-linearity to capture n-gram interactions, while keeping inference ~2–3× label-linear cost. Failure: dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item >5× label-linear.
```

```python
"""label-factorized: per-position byte embeddings summed, ReLU hidden layer, linear output.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the last 20% of train.in (tuning split)
EMBED_DIMS = [16, 32, 64]
LRS = [1e-3, 3e-3]
SEED = 1
L2 = 1e-4
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


def train_model(X, y, C, D, lr, epochs, rng):
    """Train embeddings (d,256,D), W2 (D,C), b2 (C). Returns (E, W2, b2)."""
    d = X.shape[1]
    # Embeddings: (d, 256, D)
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / D)).astype(np.float32)
    W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            # Sum embeddings: (B, D)
            hid = np.zeros((len(yb), D), np.float32)
            for j in range(d):
                hid += E[j, xb[:, j]]
            # ReLU
            hid = np.maximum(hid, 0)
            # Linear
            logits = hid @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # Backward
            gW2 = hid.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            g_hid = probs @ W2.T
            g_hid *= (hid > 0)
            gE = np.zeros_like(E)
            for j in range(d):
                # gE[j, byte, :] += g_hid[i] for each sample i where xb[i,j]==byte
                np.add.at(gE[j], xb[:, j], g_hid)
            gE += L2 * E
            grads = [gE, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return E, W2, b2


def accuracy(X, y, C, E, W2, b2):
    d = X.shape[1]
    hid = np.zeros((X.shape[0], E.shape[2]), np.float32)
    for j in range(d):
        hid += E[j, X[:, j]]
    hid = np.maximum(hid, 0)
    logits = hid @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    n = X_all.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_weights = None

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for D in EMBED_DIMS:
        for lr in LRS:
            r = np.random.default_rng(SEED)  # reset for each config
            E, W2, b2 = train_model(X_train, y_train, C, D, lr, epochs, r)
            acc = accuracy(X_val, y_val, C, E, W2, b2)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, lr)
                best_weights = (E.copy(), W2.copy(), b2.copy())

    # Retrain on full data with best config
    D, lr = best_cfg
    r = np.random.default_rng(SEED)
    E, W2, b2 = train_model(X_all, y_all, C, D, lr, epochs, r)

    # Pack weights: magic, d, D, C, then E[d][256][D], W2[D][C], b2[C]
    magic = 0x46414354  # "FACT"
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-factorized inference: ./infer weights.bin in.rec N
 * Input: 16-byte records. Output: one-byte class index per record. */
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
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W2 = E + (long)d * 256 * D;
    const float *b2 = W2 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hid = calloc(D, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(hid, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) hid[u] += e[u];
        }
        for (int u = 0; u < D; u++) if (hid[u] < 0.0f) hid[u] = 0.0f;
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < D; u++) {
            float h = hid[u];
            if (h == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += h * W2[(long)u * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```