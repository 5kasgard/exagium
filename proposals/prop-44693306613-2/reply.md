```hypothesis
Family: label-emb (embedding-based linear model). Learns a dense D-dim embedding for each of the 16×256 byte positions, then a single linear layer to 65 classes. Dense embeddings capture semantic similarity between characters (e.g., letters cluster) and generalize better than one-hot linear on small data. If it fails to beat label-linear, it suggests the 16-byte context has no learnable similarity structure beyond linear one-hot features, or that the dataset is too small to estimate embeddings reliably.
```

```python
"""label-emb: embedding + linear classifier for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses embedding dim D on tuning split (last 20%),
then trains on all data. Exports: magic, d, D, C; embeddings[d][256][D]; W[d*D][C]; bias[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
L2 = 1e-4
LR_GRID = [1e-3, 3e-3]
D_GRID = [16, 32, 64]
EPOCHS_GRID = [10, 20, 30]
TUNE_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def evaluate(X, y, emb, W, bias):
    # X: (n, d) uint8
    n, d = X.shape
    C = bias.shape[0]
    D = emb.shape[2]
    # compute logits: (n, C)
    # logits = bias + sum_j emb[j, X[:,j], :] @ W[j*D:(j+1)*D, :]
    logits = np.tile(bias, (n, 1))
    for j in range(d):
        e = emb[j, X[:, j]]  # (n, D)
        logits += e @ W[j*D:(j+1)*D]
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X_tr, y_tr, X_te, y_te, D, lr, epochs, rng):
    n, d = X_tr.shape
    C = 65
    # init embeddings: (d, 256, D)
    emb = rng.normal(0, 1/math.sqrt(D), (d, 256, D)).astype(np.float32)
    # init W: (d*D, C)
    W = rng.normal(0, 1/math.sqrt(d*D), (d*D, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bsz = len(yb)
            # forward
            logits = np.tile(bias, (bsz, 1))
            emb_used = np.zeros((bsz, d, D), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]  # (bsz, D)
                emb_used[:, j] = e
                logits += e @ W[j*D:(j+1)*D]
            # grad
            g = softmax_cross_entropy_grad(logits, yb)  # (bsz, C)
            # gradients
            g_bias = g.sum(axis=0)
            g_W = np.zeros_like(W)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                e = emb_used[:, j]  # (bsz, D)
                g_W[j*D:(j+1)*D] = e.T @ g + L2 * W[j*D:(j+1)*D]
                # gradient wrt embeddings: g @ W[j*D:(j+1)*D].T
                g_e = g @ W[j*D:(j+1)*D].T  # (bsz, D)
                # scatter add to g_emb[j, xb[:,j]]
                np.add.at(g_emb[j], xb[:, j], g_e)
            g_emb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [(emb, g_emb, m_emb, v_emb),
                                      (W, g_W, m_W, v_W),
                                      (bias, g_bias, m_b, v_b)]:
                m *= beta1; m += (1-beta1)*grad
                v *= beta2; v += (1-beta2)*grad*grad
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    acc = evaluate(X_te, y_te, emb, W, bias)
    return emb, W, bias, acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    n_tr = int(n * (1 - TUNE_FRAC))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_weights = None
    for D in D_GRID:
        for lr in LR_GRID:
            for epochs in EPOCHS_GRID:
                # deterministic per config
                config_rng = np.random.default_rng(SEED + hash((D, lr, epochs)) % 2**32)
                emb, W, bias, acc = train_model(X_tr, y_tr, X_te, y_te, D, lr, epochs, config_rng)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (D, lr, epochs)
                    best_weights = (emb, W, bias)
    print(f"Best: D={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, tune_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data with best params
    D, lr, epochs = best_params
    full_rng = np.random.default_rng(SEED + 12345)
    emb, W, bias, _ = train_model(X, y, X, y, D, lr, epochs, full_rng)  # tune set same as train for final
    # pack weights
    magic = 0x454D424C  # 'LBME'
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=16). Output: one byte class index per record.
 * Weights format: magic(0x454D424C), d, D, C (int32 LE); embeddings[d][256][D] (float32 LE);
 * W[d*D][C] (float32 LE); bias[C] (float32 LE). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d <= 0 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *bias = W + (long)d * D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(logits, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            const float *Wj = W + (long)j * D * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int u = 0; u < D; u++) sum += e[u] * Wj[(long)u * C + c];
                logits[c] += sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```