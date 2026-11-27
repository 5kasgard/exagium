```hypothesis
We propose a low-rank factorized linear model (label-lowrank). The linear baseline (label-linear) learns a full 16×256×65 lookup table (266K parameters), which may overfit on 24K training examples. By factorizing each position's 256×65 matrix into a 256×D embedding and a D×65 projection (D < 65), we impose a low-rank inductive bias that captures character similarities (e.g., vowels, consonants) and position-specific patterns, improving generalization. The embedding dimension D is chosen on the tuning split (last 20% of train). If dev accuracy does not exceed label-linear's 0.3375, the low-rank constraint does not help for this task.
```

```python
"""Low-rank factorized linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1, single thread). Factorizes the per-position token-to-logit table
into embeddings (256 x D) and projection (D x C) with D chosen on tuning split.
Exported in the same format as label-linear (magic 0x54424C31, H=0) so the same infer.c works."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_D = [8, 16, 32, 64]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(X_tr, y_tr, X_val, y_val, d, D, C, rng):
    n_tr = X_tr.shape[0]
    # Initialize parameters
    Emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    W = rng.normal(0, math.sqrt(2.0 / D), (d, D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [Emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bs = len(yb)

            # Forward
            logits = np.zeros((bs, C), np.float32)
            for j in range(d):
                emb_j = Emb[j, xb[:, j]]          # (bs, D)
                logits += emb_j @ W[j]            # (bs, C)
            logits += b

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Gradients
            grad_Emb = np.zeros_like(Emb)
            grad_W = np.zeros_like(W)
            grad_b = grad_logits.sum(axis=0)

            for j in range(d):
                emb_j = Emb[j, xb[:, j]]          # (bs, D)
                grad_W[j] = emb_j.T @ grad_logits + L2 * W[j]
                grad_emb_j = grad_logits @ W[j].T  # (bs, D)
                # Scatter back to Emb[j]
                np.add.at(grad_Emb[j], xb[:, j], grad_emb_j)
            grad_Emb += L2 * Emb

            # Adam update
            t += 1
            grads = [grad_Emb, grad_W, grad_b]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Validation accuracy
    logits_val = np.zeros((X_val.shape[0], C), np.float32)
    for j in range(d):
        logits_val += Emb[j, X_val[:, j]] @ W[j]
    logits_val += b
    return accuracy(logits_val, y_val), Emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning (file order)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = CANDIDATE_D[0], -1.0
    best_params = None

    for D in CANDIDATE_D:
        acc, Emb, W, b = train_model(X_tr, y_tr, X_val, y_val, d, D, C, rng)
        if acc > best_acc:
            best_acc, best_D = acc, D
            best_params = (Emb, W, b)

    # Retrain on full data with best D
    rng = np.random.default_rng(SEED)
    _, Emb, W, b = train_model(X, y, X[:1], y[:1], d, best_D, C, rng)  # dummy val

    # Build full table T (d, 256, C) = Emb @ W
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        T[j] = Emb[j] @ W[j]

    # Write weights in label-linear format (magic, d, H=0, C, T, b)
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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