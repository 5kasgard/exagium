```hypothesis
Family: label-mlp2 (two-hidden-layer MLP with per-position byte embeddings).
Why: The task is character-level language modeling; a deeper network can capture longer-range dependencies in the 16-byte context better than linear or single-layer models, while still being fast at inference due to precomputed lookup tables.
Failure: If dev accuracy does not exceed label-linear (0.3375) or inference cost exceeds label-gbdt (131k Ir/item), the model is not an improvement.
```

```python
"""Two-hidden-layer MLP with per-position byte embeddings (label-mlp2).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported format: magic 0x4D4C5032, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (H1, H2)
CONFIGS = [(64, 32), (128, 64), (256, 128)]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_positions(X, d, fit_vals=None):
    """Return list of one-hot matrices E_j (256 x n_vals) and encoded Z (n x total_vals)."""
    enc = []
    if fit_vals is None:
        fit_vals = [np.unique(X[:, j]) for j in range(d)]
    for j in range(d):
        vals = fit_vals[j]
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z, fit_vals


def train_model(Z, y, k, H1, H2, C, rng, epochs):
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = Z.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g3 = (probs / len(yy)).astype(np.float32)
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(0)
            gh2 = g3 @ W3.T * (a2 > 0)
            grad_W2 = h1.T @ gh2 + L2 * W2
            grad_b2 = gh2.sum(0)
            gh1 = gh2 @ W2.T * (a1 > 0)
            grad_W1 = z.T @ gh1 + L2 * W1
            grad_b1 = gh1.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(Z, y, params):
    W1, b1, W2, b2, W3, b3 = params
    a1 = Z @ W1 + b1
    h1 = np.maximum(a1, 0)
    a2 = h1 @ W2 + b2
    h2 = np.maximum(a2, 0)
    logits = h2 @ W3 + b3
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]
    split = int(0.8 * n_all)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    # Encode training split to get value sets
    enc_train, Z_train, fit_vals = encode_positions(X_train, d)
    _, Z_tune, _ = encode_positions(X_tune, d, fit_vals=fit_vals)
    k = Z_train.shape[1]

    best_acc = -1.0
    best_config = None
    best_params = None
    rng = np.random.default_rng(SEED)

    for H1, H2 in CONFIGS:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Z_train.shape[0]))
        params = train_model(Z_train, y_train, k, H1, H2, C, rng, epochs)
        acc = evaluate(Z_tune, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_config = (H1, H2)
            best_params = params

    H1, H2 = best_config
    # Retrain on full data
    enc_full, Z_full, _ = encode_positions(X_all, d)
    k_full = Z_full.shape[1]
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / Z_full.shape[0]))
    W1, b1, W2, b2, W3, b3 = train_model(Z_full, y_all, k_full, H1, H2, C, rng, epochs)

    # Build per-position lookup tables for first layer
    T1 = np.zeros((d, 256, H1), dtype=np.float32)
    offset = 0
    for j, E in enumerate(enc_full):
        T1[j] = E @ W1[offset:offset + E.shape[1], :]
        offset += E.shape[1]

    magic = 0x4D4C5032  # "MLP2"
    header = struct.pack("<5i", magic, d, H1, H2, C)
    out = header
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp2 inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x4D4C5032), d, H1, H2, C; float32 T1[d*256*H1], b1[H1], W2[H1*H2], b2[H2], W3[H2*C], b3[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (d != 16 || C != 65 || H1 <= 0 || H2 <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected_floats = (long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);
    if (!acc1 || !acc2 || !scores) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, sizeof(float) * H1);
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        memcpy(acc2, b2, sizeof(float) * H2);
        for (int h = 0; h < H1; h++) {
            float a = acc1[h];
            if (a == 0.0f) continue;
            const float *w = W2 + (long)h * H2;
            for (int h2 = 0; h2 < H2; h2++) acc2[h2] += a * w[h2];
        }
        for (int h2 = 0; h2 < H2; h2++) if (acc2[h2] < 0.0f) acc2[h2] = 0.0f;

        memcpy(scores, b3, sizeof(float) * C);
        for (int h2 = 0; h2 < H2; h2++) {
            float a = acc2[h2];
            if (a == 0.0f) continue;
            const float *w = W3 + (long)h2 * C;
            for (int c = 0; c < C; c++) scores[c] += a * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```