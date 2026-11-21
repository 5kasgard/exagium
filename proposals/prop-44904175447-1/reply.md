```hypothesis
Family: label-mlp (single hidden-layer MLP with ReLU, z-scored ordinal inputs).
Why: The 64 ordinal pixel features of optdigits have non-linear interactions that a linear model cannot capture. A small MLP (H=16..64) adds modest compute (64*H + H*10 MACs/item) while greatly increasing capacity. On the tuning split we select H to maximize accuracy; the final model trains on all data. This should surpass label-linear (0.9613) and approach label-mlp16 (0.9770) at similar or lower cost.
Failure signal: dev accuracy ≤ 0.9613 or inference instructions/item > 2× label-linear.
```

```python
"""label-mlp: single hidden-layer MLP for ordinal byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, stdlib+numpy only.
Weights format: magic=0x4D4C5031, d, H, C; float32 mean[d], std[d], W1[d*H], b1[H], W2[H*C], b2[C] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [16, 32, 64]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def zscore_fit(X):
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return ((X.astype(np.float32) - mu) / sd).astype(np.float32)


def train_mlp(X, y, H, rng, epochs):
    n, d = X.shape
    C = 10
    # Xavier init
    W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backprop
            gh = g @ W2.T
            gh[a <= 0] = 0.0
            grads = [
                z.T @ gh + L2 * W1,
                gh.sum(axis=0),
                h.T @ g + L2 * W2,
                g.sum(axis=0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    W1, b1, W2, b2 = params
    a = X @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = len(X_all)
    n_tune = n_total // 5
    X_train, X_tune = X_all[:-n_tune], X_all[-n_tune:]
    y_train, y_tune = y_all[:-n_tune], y_all[-n_tune:]

    # Preprocess on train split
    mu, sd = zscore_fit(X_train)
    X_train_z = zscore_apply(X_train, mu, sd)
    X_tune_z = zscore_apply(X_tune, mu, sd)

    rng = np.random.default_rng(SEED)
    best_H, best_acc = H_CHOICES[0], -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for H in H_CHOICES:
        # deterministic retrain for each H
        rng_h = np.random.default_rng(SEED + H)
        params = train_mlp(X_train_z, y_train, H, rng_h, epochs)
        acc = evaluate(X_tune_z, y_tune, params)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on full data with best_H
    mu_full, sd_full = zscore_fit(X_all)
    X_all_z = zscore_apply(X_all, mu_full, sd_full)
    rng_full = np.random.default_rng(SEED + best_H)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_total))
    W1, b1, W2, b2 = train_mlp(X_all_z, y_all, best_H, rng_full, epochs_full)

    # Write weights.bin
    d, H, C = X_all.shape[1], best_H, card["classes"]
    magic = 0x4D4C5031  # 'MLP1'
    header = struct.pack("<4I", magic, d, H, C)
    # mean, std, W1, b1, W2, b2 all float32 little-endian
    out = bytearray()
    out += header
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   uint32_t magic=0x4D4C5031, d, H, C;
 *   float mean[d], std[d], W1[d*H], b1[H], W2[H*C], b2[C] (row-major).
 * Input record: d bytes. Output: one byte class index per record. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x4D4C5031) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16
        + (size_t)d * 4 * 2                // mean, std
        + (size_t)d * H * 4                // W1
        + (size_t)H * 4                    // b1
        + (size_t)H * C * 4                // W2
        + (size_t)C * 4;                   // b2
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std  = mean + d;
    const float *W1   = std  + d;
    const float *b1   = W1   + (size_t)d * H;
    const float *W2   = b1   + H;
    const float *b2   = W2   + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score + first layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d; j++) {
                float xz = (xi[j] - mean[j]) / std[j];
                sum += xz * W1[(size_t)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }
        // second layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            scores[c] = sum;
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```