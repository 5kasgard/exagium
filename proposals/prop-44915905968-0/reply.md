```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which is crucial for tabular data like Adult where combinations of categorical features (e.g., education × occupation) are predictive. With a small embedding size (K=8), the model stays fast: inference is O(d·K) per item (d=14). We expect dev accuracy >0.8524 (label-linear) at similar or lower latency. If accuracy ≤0.8524 or instructions/item exceed label-linear by >20%, the hypothesis fails.
```

```python
"""label-fm: Factorization Machine for tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Trains FM with Adam on last 20% tuning split to pick K, then retrains on all data.
Exports: magic=0x464D0001, d, K, C=2; float32 bias, linear[d*256], emb[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned offline on tuning split)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
K_CANDIDATES = [4, 8, 16]  # embedding sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, bias, linear, emb, K):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    # linear term
    lin = bias + np.sum(linear[X, np.arange(d)], axis=1)  # linear[d,256] -> need proper indexing
    # Actually linear is (d, 256). We'll compute per feature.
    lin = bias * np.ones(n, dtype=np.float32)
    for j in range(d):
        lin += linear[j, X[:, j]]
    # interaction term: 0.5 * ( (sum emb)^2 - sum emb^2 )
    sum_emb = np.zeros((n, K), dtype=np.float32)
    sum_emb_sq = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        e = emb[j, X[:, j], :]  # (n, K)
        sum_emb += e
        sum_emb_sq += e * e
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    return lin + inter


def train_fm(X, y, K, rng, epochs=EPOCHS, batch=BATCH, lr=LR, l2=L2):
    n, d = X.shape
    # init
    bias = 0.0
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # forward
            lin = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (bsz,)
            sum_emb = np.zeros((bsz, K), dtype=np.float32)
            sum_emb_sq = np.zeros((bsz, K), dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j], :]
                sum_emb += e
                sum_emb_sq += e * e
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            logits = lin + inter
            p = sigmoid(logits)
            # gradient of BCE loss
            grad = (p - yb).astype(np.float32)  # (bsz,)
            # bias grad
            g_bias = grad.mean()
            # linear grad
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            g_lin /= bsz
            g_lin += l2 * linear
            # emb grad
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # dL/demb_j = grad * (sum_emb - emb_j)  [since d(inter)/demb_j = sum_emb - emb_j]
                diff = (sum_emb - emb[j, xb[:, j], :])  # (bsz, K)
                g_emb[j, xb[:, j], :] += (grad[:, None] * diff)
            g_emb /= bsz
            g_emb += l2 * emb
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return bias, linear, emb


def evaluate(X, y, bias, linear, emb, K):
    logits = fm_forward(X, bias, linear, emb, K)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    n_tune = n // 5
    n_train = n - n_tune
    # tuning split: last 20% in file order
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    best_K = K_CANDIDATES[0]
    best_acc = -1.0
    for K in K_CANDIDATES:
        bias, linear, emb = train_fm(X_train, y_train, K, np.random.default_rng(SEED))
        acc = evaluate(X_tune, y_tune, bias, linear, emb, K)
        if acc > best_acc:
            best_acc = acc
            best_K = K

    # retrain on all data with best_K
    bias, linear, emb = train_fm(X, y, best_K, np.random.default_rng(SEED))

    # write weights
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D0001, d, best_K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0 or 1).
 * Weights format: magic(0x464D0001), d, K, C=2; float32 bias, linear[d*256], emb[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = *bias;
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_emb_sq, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_emb[kk] * sum_emb[kk] - sum_emb_sq[kk];
        }
        inter *= 0.5f;

        float logit = lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```