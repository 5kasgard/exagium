```hypothesis
Family: label-mlptiny (single-hidden-layer MLP with per-pixel lookup tables, H∈{4,8,16,32} chosen on the 20% tuning split).
Why: The linear model (label-linear) ignores pixel interactions; a tiny ReLU layer adds non-linearity at modest cost. Optdigits has only 64 ordinal pixels (0–16), so a small hidden layer should capture useful combinations without overfitting. Hyperparameter search on the tuning split picks the best capacity.
Failure indication: Dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ label-linear's 5059 Ir/item.
```

```python
import json, math, struct, sys
import numpy as np

H_CHOICES = [4, 8, 16, 32]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4D4C5054  # 'MLPT'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_ordinal(X, j, mu=None, sd=None):
    v = X[:, j].astype(np.float64)
    if mu is None:
        mu = v.mean()
    if sd is None:
        sd = v.std() or 1.0
    E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
    return E, mu, sd

def train_model(X, y, H, encodings, epochs, rng):
    d = X.shape[1]
    C = int(y.max()) + 1
    Z = np.concatenate([encodings[j][X[:, j]] for j in range(d)], axis=1)
    n = Z.shape[0]
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
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yy)), yy] -= 1.0
            grad_logits /= len(yy)
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T
            grad_a = grad_h * (a > 0)
            grad_W1 = z.T @ grad_a + L2 * W1
            grad_b1 = grad_a.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def accuracy(X, y, params, encodings):
    W1, b1, W2, b2 = params
    d = X.shape[1]
    Z = np.concatenate([encodings[j][X[:, j]] for j in range(d)], axis=1)
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = np.argmax(logits, axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n_all = X_all.shape[0]
    split_idx = int(n_all * 0.8)
    X_train, X_tune = X_all[:split_idx], X_all[split_idx:]
    y_train, y_tune = y_all[:split_idx], y_all[split_idx:]

    encodings = []
    for j in range(d):
        E, _, _ = encode_ordinal(X_train, j)
        encodings.append(E)

    best_H = None
    best_acc = -1.0
    best_params = None
    rng = np.random.default_rng(SEED)

    for H in H_CHOICES:
        model_rng = np.random.default_rng(SEED + H)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / X_train.shape[0]))
        params = train_model(X_train, y_train, H, encodings, epochs, model_rng)
        acc = accuracy(X_tune, y_tune, params, encodings)
        if acc > best_acc:
            best_acc = acc
            best_H = H
            best_params = params

    encodings_full = []
    for j in range(d):
        E, _, _ = encode_ordinal(X_all, j)
        encodings_full.append(E)
    final_rng = np.random.default_rng(SEED + best_H)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    final_params = train_model(X_all, y_all, best_H, encodings_full, epochs_full, final_rng)

    W1, b1, W2, b2 = final_params
    T1_list = []
    for j in range(d):
        T1_list.append((encodings_full[j] @ W1[j:j+1]).astype(np.float32))
    T1 = np.stack(T1_list)

    out = struct.pack("<4i", MAGIC, d, best_H, C)
    out += T1.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4D4C5054) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected_floats = (long)d * 256 * H + H + (long)H * C + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T1 = (const float *)(hd + 4);
    const float *b1 = T1 + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H);
    float *acc2 = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc1[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;
        for (int c = 0; c < C; c++) acc2[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc1[h];
            for (int c = 0; c < C; c++) acc2[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc2[c] > acc2[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```