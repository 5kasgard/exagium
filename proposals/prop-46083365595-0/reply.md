```hypothesis
Family: label-fm (Factorization Machine). Adult has 14 tabular features with mixed ordinal/categorical types; pairwise interactions (e.g., education × occupation, capital-gain × hours) are known to boost accuracy over linear models. FM captures all pairwise interactions via low-rank embeddings (size k), adding only O(d*256*k) parameters. With k=8, it should beat label-linear (0.8524) while keeping inference fast (~few hundred ns/item). If dev accuracy ≤ 0.8524 or latency ≥ label-linear, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the 14 features as a field with 256 categories (quantile bins or category indices).
Learns linear weights w0 + w_{j,v} and embeddings e_{j,v} in R^k.
Prediction: w0 + sum_j w_{j,x_j} + 0.5 * ( (sum_j e_{j,x_j})^2 - sum_j e_{j,x_j}^2 ).
Trained with Adam on logistic loss. Hyperparameters chosen on last 20% of train.in (tuning split).
Exported as binary: magic 0x464D4C31, d, k, C; float32 w0, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-2, 3e-3, 1e-3]
L2_CHOICES = [1e-4, 1e-5]
EPOCHS_CHOICES = [20, 40, 60]
BATCH = 256
SAMPLES = 600000  # max samples per epoch for early stopping


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class FM:
    def __init__(self, d, k, C, rng):
        self.d = d
        self.k = k
        self.C = C
        self.w0 = 0.0
        self.linear = np.zeros((d, 256), dtype=np.float32)
        self.emb = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)

    def forward(self, X):
        """X: (n, d) uint8. Returns logits (n, C)."""
        n = X.shape[0]
        # Linear part
        lin = self.w0 + np.sum(self.linear[np.arange(self.d), X.T], axis=0)  # (n,)
        # Interaction part
        # Sum embeddings per sample: (n, k)
        sum_emb = np.zeros((n, self.k), dtype=np.float32)
        sum_emb_sq = np.zeros((n, self.k), dtype=np.float32)
        for j in range(self.d):
            e = self.emb[j, X[:, j]]  # (n, k)
            sum_emb += e
            sum_emb_sq += e * e
        inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)  # (n,)
        logits = lin + inter
        return logits[:, None]  # (n, 1) for binary C=2, we use single logit for class 1

    def predict_proba(self, X):
        logits = self.forward(X)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.hstack([1 - p1, p1])

    def sgd_step(self, X, y, lr, l2, adam_state, t):
        """One minibatch update with Adam. X: (b, d), y: (b,) in {0,1}."""
        b = X.shape[0]
        # Forward
        lin = self.w0 + np.sum(self.linear[np.arange(self.d), X.T], axis=0)
        sum_emb = np.zeros((b, self.k), dtype=np.float32)
        sum_emb_sq = np.zeros((b, self.k), dtype=np.float32)
        for j in range(self.d):
            e = self.emb[j, X[:, j]]
            sum_emb += e
            sum_emb_sq += e * e
        inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
        logits = lin + inter
        p1 = 1.0 / (1.0 + np.exp(-logits))
        # Gradient of logistic loss: (p1 - y) for each sample
        grad = (p1 - y).astype(np.float32)  # (b,)

        # Gradients
        # w0
        gw0 = grad.mean()
        # linear
        glinear = np.zeros_like(self.linear)
        for j in range(self.d):
            vals = X[:, j]
            for v in range(256):
                mask = (vals == v)
                if np.any(mask):
                    glinear[j, v] = grad[mask].mean()
        # embeddings
        gemb = np.zeros_like(self.emb)
        for j in range(self.d):
            vals = X[:, j]
            # For each sample, contribution to embedding gradient: grad_i * (sum_emb_i - e_{j,v_i})
            # We can vectorize by accumulating per (j, v)
            for v in range(256):
                mask = (vals == v)
                if np.any(mask):
                    # sum_emb for these samples
                    se = sum_emb[mask]
                    # e is same for all these samples
                    e = self.emb[j, v]
                    # gradient for this embedding: mean over samples of grad * (se - e)
                    gemb[j, v] = np.mean(grad[mask, None] * (se - e), axis=0)

        # L2 regularization
        gw0 += l2 * self.w0
        glinear += l2 * self.linear
        gemb += l2 * self.emb

        # Adam update
        params = [self.w0, self.linear, self.emb]
        grads = [gw0, glinear, gemb]
        for idx, (p, gr) in enumerate(zip(params, grads)):
            m, v = adam_state[idx]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def evaluate(model, X, y):
    logits = model.forward(X)
    preds = (logits > 0).astype(np.int64).flatten()
    return np.mean(preds == y)


def train_fm(X_train, y_train, X_val, y_val, k, lr, l2, max_epochs, rng):
    d = X_train.shape[1]
    model = FM(d, k, 2, rng)
    # Adam state: for w0 (scalar), linear (d,256), emb (d,256,k)
    adam_state = [
        (np.zeros_like(model.w0), np.zeros_like(model.w0)),
        (np.zeros_like(model.linear), np.zeros_like(model.linear)),
        (np.zeros_like(model.emb), np.zeros_like(model.emb)),
    ]
    n = X_train.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    max_steps = min(max_epochs * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            if len(idx) == 0:
                continue
            t += 1
            model.sgd_step(X_train[idx], y_train[idx], lr, l2, adam_state, t)
            if t >= max_steps:
                break
        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = (model.w0, model.linear.copy(), model.emb.copy())
        if t >= max_steps:
            break
    # Restore best
    if best_params:
        model.w0, model.linear, model.emb = best_params
    return model, best_val


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_model = None
    best_val_acc = -1
    best_cfg = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                for epochs in EPOCHS_CHOICES:
                    # Use a fresh RNG for each config to keep determinism across configs
                    cfg_rng = np.random.default_rng(SEED + hash((k, lr, l2, epochs)) % 2**32)
                    model, val_acc = train_fm(X_train, y_train, X_val, y_val, k, lr, l2, epochs, cfg_rng)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model
                        best_cfg = (k, lr, l2, epochs)
    print(f"Best config: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, epochs={best_cfg[3]}, val_acc={best_val_acc:.4f}")

    # Retrain on full data with best config
    full_rng = np.random.default_rng(SEED)
    final_model, _ = train_fm(X, y, X, y, *best_cfg, full_rng)  # X_val not used, but function expects it

    # Save weights
    # Format: magic, d, k, C; then w0, linear[d*256], emb[d*256*k] as float32 little-endian
    magic = 0x464D4C31  # 'FM1' in ASCII? Actually 0x46='F', 0x4D='M', 0x4C='L', 0x31='1'
    out = struct.pack("<4i", magic, d, best_cfg[0], C)
    out += struct.pack("<f", final_model.w0)
    out += final_model.linear.astype("<f4").tobytes()
    out += final_model.emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index (0 or 1).
 * Weights format: uint32 magic 0x464D4C31, int32 d, int32 k, int32 C;
 * float32 w0, float32 linear[d*256], float32 emb[d*256*k]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0 || d > 100 || k > 100) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear part
        float logit = w0;
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }

        // Interaction part
        memset(sum_emb, 0, sizeof(float) * k);
        memset(sum_emb_sq, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_emb_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) {
            float s = sum_emb[u];
            inter += 0.5f * (s * s - sum_emb_sq[u]);
        }
        logit += inter;

        // Sigmoid and threshold
        float p1 = 1.0f / (1.0f + expf(-logit));
        unsigned char best = (p1 > 0.5f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```