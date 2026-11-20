```hypothesis
Family: label-embed. A position-specific embedding model: each of the 16 context positions has a learned 256×E embedding table; the 16 E-dimensional vectors are summed, passed through a ReLU hidden layer of size H, then a linear output layer to 65 classes. This captures positional n-gram statistics with far fewer parameters than one-hot linear (16×256×E + H×E + 65×H vs 16×256×65) and should generalize better on the small tinyshakespeare training set. Chosen hyperparameters (E, H) on the tuning split (last 20% of train). If validation accuracy does not exceed label-linear (0.3375) by at least 0.01, the family does not justify its complexity.
```

```python
"""label-embed: position-specific embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Grid-searches E in {4,8,16}, H in {16,32,64} on the last 20% tuning split, then retrains on all data.
Exported format (little-endian):
  int32 magic=0x454D4245, d=16, E, H, C=65
  float32 emb[d][256][E]   (row-major: position, byte, embedding)
  float32 W1[H][E], b1[H]
  float32 W2[C][H], b2[C]
"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
E_CHOICES = (4, 8, 16)
H_CHOICES = (16, 32, 64)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_tuning(X, y, frac=0.2):
    """Last 20% in file order as tuning set."""
    n = X.shape[0]
    cut = int(n * (1 - frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def init_params(d, E, H, C, rng):
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / E), (H, E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (C, H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X):
    """X: (n, d) uint8. Returns logits (n, C), hidden (n, H) for backprop."""
    n, d = X.shape
    # sum embeddings: (n, E)
    summed = np.zeros((n, E), np.float32)
    for j in range(d):
        summed += emb[j, X[:, j]]
    # hidden layer
    hidden_pre = summed @ W1.T + b1
    hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
    logits = hidden @ W2.T + b2
    return logits, hidden, hidden_pre, summed


def loss_grad(logits, y):
    """Softmax cross-entropy gradient w.r.t logits. Returns (n, C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n


def train_model(X_tr, y_tr, X_val, y_val, E, H, rng):
    n, d = X_tr.shape
    C = 65
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, hidden, hidden_pre, summed = forward(emb, W1, b1, W2, b2, Xb)
            g_logits = loss_grad(logits, yb)  # (b, C)
            # grad W2, b2
            g_W2 = g_logits.T @ hidden + L2 * W2
            g_b2 = g_logits.sum(0)
            # grad hidden
            g_hidden = g_logits @ W2
            g_hidden[hidden_pre <= 0] = 0.0
            # grad W1, b1
            g_W1 = g_hidden.T @ summed + L2 * W1
            g_b1 = g_hidden.sum(0)
            # grad summed -> grad emb
            g_summed = g_hidden @ W1  # (b, E)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # accumulate gradients for each position
                # Xb[:, j] are byte indices (b,)
                np.add.at(g_emb[j], Xb[:, j], g_summed)
            g_emb += L2 * emb
            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    logits_val, _, _, _ = forward(emb, W1, b1, W2, b2, X_val)
    preds = logits_val.argmax(1)
    acc = (preds == y_val).mean()
    return acc, (emb, W1, b1, W2, b2)


def pack_weights(d, E, H, C, params):
    emb, W1, b1, W2, b2 = params
    magic = 0x454D4245  # 'EMBE'
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (emb.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes())
    return header + body


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # split tuning set (last 20%)
    X_tr, y_tr, X_val, y_val = split_tuning(X, y, 0.2)

    best_acc, best_params, best_E, best_H = -1, None, None, None
    for E in E_CHOICES:
        for H in H_CHOICES:
            # fresh rng for each config to keep deterministic
            cfg_rng = np.random.default_rng(SEED + E * 100 + H)
            acc, params = train_model(X_tr, y_tr, X_val, y_val, E, H, cfg_rng)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H
    print(f"chosen E={best_E}, H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    full_rng = np.random.default_rng(SEED + best_E * 100 + best_H)
    _, final_params = train_model(X, y, X[:1], y[:1], best_E, best_H, full_rng)  # dummy val
    out = pack_weights(d, best_E, best_H, C, final_params)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x454D4245, d=16, E, H, C=65
 *   float emb[d][256][E]
 *   float W1[H][E], b1[H]
 *   float W2[C][H], b2[C]
 */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L
        + (long)d * 256 * E
        + (long)H * E + H
        + (long)C * H + C;
    if (nw != 4 * expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)H * E;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* sum embeddings */
        for (int e = 0; e < E; e++) summed[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) summed[e] += eptr[e];
        }
        /* hidden = ReLU(summed @ W1.T + b1) */
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            const float *w = W1 + (long)h * E;
            for (int e = 0; e < E; e++) acc += summed[e] * w[e];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }
        /* logits = hidden @ W2.T + b2 */
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            const float *w = W2 + (long)c * H;
            for (int h = 0; h < H; h++) acc += hidden[h] * w[h];
            logits[c] = acc;
        }
        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```