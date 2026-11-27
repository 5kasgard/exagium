```hypothesis
We propose a small MLP with one hidden layer (size chosen from {4,8,16,32} on the tuning split) and per-pixel z-score normalization. The 8x8 digit images benefit from non-linear feature combinations that a single ReLU layer can capture, while the model stays tiny (≈64×H + H×10 weights). If the dev accuracy does not exceed label-linear's 0.9613, the family does not beat the baseline.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_H = [4, 8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 64

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Normalization stats on training portion
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    def normalize(X_part):
        return (X_part.astype(np.float32) - mu) / sd

    Z_train = normalize(X_train)
    Z_val = normalize(X_val)

    best_H = None
    best_acc = -1.0
    best_params = None

    for H in CANDIDATE_H:
        rng = np.random.default_rng(SEED)
        W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        n_train = Z_train.shape[0]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        for _ in range(epochs):
            order = rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s:s + BATCH]
                z, yy = Z_train[idx], y_train[idx]
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(len(yy)), yy] -= 1.0
                g = probs / len(yy)
                grad_W2 = h.T @ g + L2 * W2
                grad_b2 = g.sum(0)
                gh = g @ W2.T
                gh[a <= 0] = 0
                grad_W1 = z.T @ gh + L2 * W1
                grad_b1 = gh.sum(0)
                grads = [grad_W1, grad_b1, grad_W2, grad_b2]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        h_val = np.maximum(Z_val @ W1 + b1, 0)
        logits_val = h_val @ W2 + b2
        pred = logits_val.argmax(axis=1)
        acc = (pred == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H
            best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    print(f"Selected H={best_H} with val acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset with best_H
    mu_full = X.mean(axis=0).astype(np.float32)
    sd_full = X.std(axis=0).astype(np.float32)
    sd_full[sd_full == 0] = 1.0
    Z_full = (X.astype(np.float32) - mu_full) / sd_full

    rng = np.random.default_rng(SEED)
    W1 = (rng.uniform(-1, 1, (d, best_H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(best_H, np.float32)
    W2 = (rng.uniform(-1, 1, (best_H, C)) * math.sqrt(6.0 / best_H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_full = Z_full.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_full[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = probs / len(yy)
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh[a <= 0] = 0
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic, d, H, C; mean[d], std[d]; W1[d*H], b1[H], W2[H*C], b2[C] (all float32 LE)
    magic = 0x4D4C5031  # 'MLP1'
    out = struct.pack("<4i", magic, d, best_H, C)
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
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 8L * d + 4L * (d * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W1 = std + d;
    const float *b1 = W1 + (long)d * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);
    float *norm = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            norm[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < d; j++) sum += norm[j] * W1[(long)j * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            output[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```