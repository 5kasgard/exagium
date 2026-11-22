```hypothesis
Family: label-emb (per-position byte embeddings + linear head). TinyShakespeare characters have strong local similarity (e.g., vowels, punctuation clusters); embeddings capture this with far fewer parameters than one-hot. A 16-D embedding per position (16*256*16 = 65k params) plus a linear layer (16*16*65 = 17k) is smaller and faster than label-linear's 4096*65 = 266k weights. If dev accuracy ≤ 0.3375 or latency ≥ label-linear, the hypothesis fails.
```
```python
"""label-emb: per-position byte embeddings + linear classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Chooses embedding dim D on last 20% of train (tuning), then retrains on all data.
Exports: magic 0x454D424C, d=16, D, C=65; embeddings[16][256][D] (float32), W[16*D][65], b[65]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [8, 16, 32, 64]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_crossentropy_grad(logits, y):
    """Return gradient of CE w.r.t logits, shape (B, C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_one(X, y, X_val, y_val, D, rng, epochs):
    """Train embeddings+linear on (X,y), validate on (X_val,y_val). Return best weights and val acc."""
    n, d = X.shape
    C = 65
    # params: emb[16][256][D], W[16*D][C], b[C]
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_acc, best_weights = -1.0, None

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)
            # forward: gather embeddings
            emb_concat = np.empty((B, d * D), np.float32)
            for j in range(d):
                emb_concat[:, j * D:(j + 1) * D] = emb[j, xb[:, j]]
            logits = emb_concat @ W + b
            # loss grad
            g_logits = softmax_crossentropy_grad(logits, yb).astype(np.float32)
            # gradients
            g_W = emb_concat.T @ g_logits + L2 * W
            g_b = g_logits.sum(axis=0)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                g_emb[j, xb[:, j]] += g_logits @ W[j * D:(j + 1) * D].T
            g_emb += L2 * emb
            grads = [g_emb, g_W, g_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # validation accuracy
        if X_val is not None:
            emb_concat_val = np.empty((len(X_val), d * D), np.float32)
            for j in range(d):
                emb_concat_val[:, j * D:(j + 1) * D] = emb[j, X_val[:, j]]
            logits_val = emb_concat_val @ W + b
            pred = logits_val.argmax(axis=1)
            acc = (pred == y_val).mean()
            if acc > best_acc:
                best_acc = acc
                best_weights = [p.copy() for p in params]
    if best_weights is None:
        best_weights = params
        # compute final val acc
        emb_concat_val = np.empty((len(X_val), d * D), np.float32)
        for j in range(d):
            emb_concat_val[:, j * D:(j + 1) * D] = emb[j, X_val[:, j]]
        logits_val = emb_concat_val @ W + b
        best_acc = (logits_val.argmax(axis=1) == y_val).mean()
    return best_weights, best_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65
    rng = np.random.default_rng(SEED)

    n = len(X)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    # Hyperparameter search on tuning split
    best_D, best_val_acc = None, -1.0
    for D in D_CANDIDATES:
        weights, val_acc = train_one(X_train, y_train, X_tune, y_tune, D, rng, epochs)
        if val_acc > best_val_acc:
            best_val_acc, best_D = val_acc, D
            best_weights_tune = weights
    print(f"chosen D={best_D} tuning_acc={best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data with chosen D (same epochs formula on full n)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_weights, _ = train_one(X, y, None, None, best_D, rng, epochs_full)
    emb, W, b = final_weights

    # Pack weights: magic, d, D, C; then emb[16][256][D], W[16*D][C], b[C] (all float32 LE)
    out = struct.pack("<4i", 0x454D424C, d, best_D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (0..64).
 * Weights format: magic 0x454D424C, int32 d=16, D, C=65;
 *   float32 emb[d][256][D] (row-major: position, byte, dim),
 *   float32 W[d*D][C], float32 b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b, sizeof(float) * C);  // start with bias (will add linear part later)
        // Actually we need to compute emb_concat then matvec. Do in two steps:
        // 1. gather embeddings into a temporary buffer of size d*D
        float *emb_concat = malloc(sizeof(float) * d * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(emb_concat + j * D, e, sizeof(float) * D);
        }
        // 2. scores = emb_concat @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * D; idx++) {
            float v = emb_concat[idx];
            if (v != 0.0f) {
                const float *w = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }
        free(emb_concat);

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```