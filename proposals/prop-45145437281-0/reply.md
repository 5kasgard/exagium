```hypothesis
Family: label-fm (factorization machine with per-column embeddings).
Why: Adult has strong categorical interactions; FM captures pairwise effects in O(d*k) time (d=14, k=4) with a tiny memory footprint. It should exceed label-linear accuracy (0.8524) while staying well below label-mlp16 cost (2325 Ir/item).
Failure: Dev accuracy ≤ 0.8524 or instructions/item > 1500 (2× label-linear).
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all.
Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias; float32 linear[d*256]; float32 emb[d*256*k]."""
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning
GRID = [
    {"k": 4, "lr": 0.03, "l2": 1e-4, "epochs": 20},
    {"k": 4, "lr": 0.01, "l2": 1e-4, "epochs": 20},
    {"k": 4, "lr": 0.03, "l2": 1e-3, "epochs": 20},
    {"k": 4, "lr": 0.01, "l2": 1e-3, "epochs": 20},
    {"k": 8, "lr": 0.03, "l2": 1e-4, "epochs": 15},
    {"k": 8, "lr": 0.01, "l2": 1e-4, "epochs": 15},
    {"k": 8, "lr": 0.03, "l2": 1e-3, "epochs": 15},
    {"k": 8, "lr": 0.01, "l2": 1e-3, "epochs": 15},
]
SEED = 1
BATCH = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_logit(X, bias, lin, emb, d, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n = X.shape[0]
    logits = np.full(n, bias, dtype=np.float32)
    # linear part
    for j in range(d):
        logits += lin[j, X[:, j]]
    # FM interaction part
    # sum of embeddings per example
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_emb_sq = np.zeros(n, dtype=np.float32)
    for j in range(d):
        e = emb[j, X[:, j], :]          # (n, k)
        sum_emb += e
        sum_emb_sq += np.sum(e * e, axis=1)
    dot_sum = np.sum(sum_emb * sum_emb, axis=1)
    logits += 0.5 * (dot_sum - sum_emb_sq)
    return logits

def accuracy(X, y, bias, lin, emb, d, k):
    logits = predict_logit(X, bias, lin, emb, d, k)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def train_fm(X, y, d, k, lr, l2, epochs, rng):
    n = X.shape[0]
    # init
    bias = np.float32(0.0)
    lin = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    emb = rng.uniform(-0.01, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)
            # forward
            logits = np.full(bsz, bias, dtype=np.float32)
            for j in range(d):
                logits += lin[j, xb[:, j]]
            sum_emb = np.zeros((bsz, k), dtype=np.float32)
            sum_emb_sq = np.zeros(bsz, dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j], :]
                sum_emb += e
                sum_emb_sq += np.sum(e * e, axis=1)
            dot_sum = np.sum(sum_emb * sum_emb, axis=1)
            logits += 0.5 * (dot_sum - sum_emb_sq)
            # gradient of logistic loss
            p = sigmoid(logits)
            grad = (p - yb).astype(np.float32)  # (bsz,)
            # bias grad
            g_bias = grad.mean()
            # linear grads
            g_lin = np.zeros_like(lin)
            for j in range(d):
                for val in range(256):
                    mask = (xb[:, j] == val)
                    if np.any(mask):
                        g_lin[j, val] = grad[mask].mean()
            # embedding grads
            g_emb = np.zeros_like(emb)
            # For each example, dlogit/demb_j = grad_i * (sum_emb_i - emb_ji)
            for j in range(d):
                e = emb[j, xb[:, j], :]          # (bsz, k)
                # (sum_emb - e) shape (bsz, k)
                diff = sum_emb - e
                # grad_i * diff_i for each i, then average over batch
                for val in range(256):
                    mask = (xb[:, j] == val)
                    if np.any(mask):
                        g_emb[j, val] = (grad[mask, None] * diff[mask]).mean(axis=0)
            # L2 regularization
            g_lin += l2 * lin
            g_emb += l2 * emb
            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            lin -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return bias, lin, emb

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng_base = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for cfg in GRID:
        rng = np.random.default_rng(rng_base.integers(2**32))
        bias, lin, emb = train_fm(X_train, y_train, d, cfg["k"], cfg["lr"], cfg["l2"], cfg["epochs"], rng)
        acc = accuracy(X_tune, y_tune, bias, lin, emb, d, cfg["k"])
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (bias, lin, emb)

    # retrain on full data with best config
    rng = np.random.default_rng(SEED)
    bias, lin, emb = train_fm(X, y, d, best_cfg["k"], best_cfg["lr"], best_cfg["l2"], best_cfg["epochs"], rng)

    # write weights.bin
    k = best_cfg["k"]
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, k, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(np.array([bias], dtype="<f4").tobytes())
        f.write(lin.astype("<f4").tobytes())
        f.write(emb.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 32) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(wb + 16);
    const float *lin = bias + 1;
    const float *emb = lin + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * k);
    if (!sum_emb) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float logit = bias[0];
        memset(sum_emb, 0, sizeof(float) * k);
        float sum_emb_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            logit += lin[j * 256 + val];
            const float *e = emb + (j * 256 + val) * k;
            for (int l = 0; l < k; l++) {
                sum_emb[l] += e[l];
            }
            float sq = 0.0f;
            for (int l = 0; l < k; l++) sq += e[l] * e[l];
            sum_emb_sq += sq;
        }
        float dot_sum = 0.0f;
        for (int l = 0; l < k; l++) dot_sum += sum_emb[l] * sum_emb[l];
        logit += 0.5f * (dot_sum - sum_emb_sq);

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_emb);
    return 0;
}
```