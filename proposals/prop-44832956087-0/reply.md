```hypothesis
Family: label-fm (factorization machine with class-specific embeddings).
Why: Adult has mixed ordinal/categorical features; FM captures pairwise interactions with low-rank embeddings, improving over linear models while staying fast and small. The 14 features × 256 values × 2 classes × k=4 factors yields ~36K parameters, inference ~O(d·k·C) operations.
Failure signal: Dev accuracy ≤ label-linear (0.8524) or Ir/item > 2× label-linear (~1500).
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D5431, d, k, C; float32 bias[C], linear[d][256][C], emb[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
K_CHOICES = [2, 4, 8]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-5]
BATCH = 64
MAX_EPOCHS = 50
SEED = 1
SAMPLES_PER_EPOCH = 600000  # max samples per epoch (like label-linear)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, k, rng):
    bias = np.zeros(C, np.float32)
    linear = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, C, k)).astype(np.float32)
    return bias, linear, emb


def forward(X, bias, linear, emb):
    """X: (n, d) uint8. Returns scores (n, C)."""
    n, d = X.shape
    C = bias.shape[0]
    k = emb.shape[3]
    scores = np.tile(bias, (n, 1))  # (n, C)
    # linear part
    for j in range(d):
        scores += linear[j, X[:, j], :]  # (n, C)
    # interaction part
    for c in range(C):
        sum_emb = np.zeros((n, k), np.float32)
        sum_emb_sq = np.zeros((n, k), np.float32)
        for j in range(d):
            e = emb[j, X[:, j], c, :]  # (n, k)
            sum_emb += e
            sum_emb_sq += e * e
        scores[:, c] += 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    return scores


def backward(X, y, scores, bias, linear, emb, lr, l2, t, state):
    """SGD with Adam. Updates params in-place. Returns new t, state."""
    n, d = X.shape
    C = bias.shape[0]
    k = emb.shape[3]
    # softmax gradient
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # (n, C)

    # gradients
    g_bias = probs.sum(axis=0)  # (C,)
    g_linear = np.zeros_like(linear)
    g_emb = np.zeros_like(emb)

    for j in range(d):
        xj = X[:, j]  # (n,)
        # linear gradient
        for c in range(C):
            g_linear[j, xj, c] += probs[:, c]
        # embedding gradient
        for c in range(C):
            # compute sum_emb and sum_emb_sq for this class
            sum_emb = np.zeros((n, k), np.float32)
            sum_emb_sq = np.zeros((n, k), np.float32)
            for jj in range(d):
                e = emb[jj, X[:, jj], c, :]
                sum_emb += e
                sum_emb_sq += e * e
            # derivative of interaction term w.r.t. emb[j, xj, c, f]
            # d(interaction)/d(emb) = sum_emb - emb
            e = emb[j, xj, c, :]  # (n, k)
            demb = probs[:, c, None] * (sum_emb - e)  # (n, k)
            g_emb[j, xj, c, :] += demb

    # L2 regularization
    g_linear += l2 * linear
    g_emb += l2 * emb

    # Adam update
    params = [bias, linear, emb]
    grads = [g_bias, g_linear, g_emb]
    for i, (p, gr) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= 0.9
        m += 0.1 * gr
        v *= 0.999
        v += 0.001 * gr * gr
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t, state


def train_model(X, y, X_val, y_val, d, C, k, lr, l2, batch, max_epochs, rng):
    bias, linear, emb = init_params(d, C, k, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in [bias, linear, emb]]
    t = 0
    n = X.shape[0]
    steps_per_epoch = max(1, math.ceil(SAMPLES_PER_EPOCH / n))
    total_steps = max_epochs * steps_per_epoch
    best_val_acc = -1.0
    best_params = None

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            scores = forward(xb, bias, linear, emb)
            t += 1
            t, state = backward(xb, yb, scores, bias, linear, emb, lr, l2, t, state)
            if t >= total_steps:
                break
        # validation
        val_scores = forward(X_val, bias, linear, emb)
        val_pred = val_scores.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (bias.copy(), linear.copy(), emb.copy())
        if t >= total_steps:
            break
    return best_val_acc, best_params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # deterministic sub-seed per config
                sub_rng = np.random.default_rng(SEED + hash((k, lr, l2)) % 2**32)
                val_acc, params = train_model(X_train, y_train, X_val, y_val, d, C, k, lr, l2, BATCH, MAX_EPOCHS, sub_rng)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cfg = (k, lr, l2)
                    best_params = params
                    print(f"New best: k={k}, lr={lr}, l2={l2}, val_acc={val_acc:.4f}", file=sys.stderr)

    k, lr, l2 = best_cfg
    print(f"Final config: k={k}, lr={lr}, l2={l2}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    full_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_model(X, y, X_val, y_val, d, C, k, lr, l2, BATCH, MAX_EPOCHS, full_rng)
    bias, linear, emb = final_params

    # write weights.bin
    magic = 0x464D5431  # 'FMT1'
    total_floats = C + d * 256 * C + d * 256 * C * k
    expected_size = 16 + 4 * total_floats
    out = struct.pack("<4i", magic, d, k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    assert len(out) == expected_size, f"size mismatch {len(out)} != {expected_size}"
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x464D5431) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32) { fprintf(stderr, "bad header\n"); return 3; }
    long total_floats = C + (long)d * 256 * C + (long)d * 256 * C * k;
    if (nw != 16 + 4 * total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * C * k);
    float *sum_emb_sq = malloc(sizeof(float) * C * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // linear part
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // interaction part
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < k; f++) {
                sum_emb[c * k + f] = 0.0f;
                sum_emb_sq[c * k + f] = 0.0f;
            }
        }
        for (int j = 0; j < d; j++) {
            const float *e = emb + (((long)j * 256 + xi[j]) * C * k);
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < k; f++) {
                    float val = e[c * k + f];
                    sum_emb[c * k + f] += val;
                    sum_emb_sq[c * k + f] += val * val;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < k; f++) {
                float se = sum_emb[c * k + f];
                float sesq = sum_emb_sq[c * k + f];
                inter += 0.5f * (se * se - sesq);
            }
            scores[c] += inter;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```