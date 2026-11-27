```hypothesis
Family: label-fm (factorization machine with binary features). The Adult dataset mixes ordinal and categorical features; a factorization machine captures pairwise interactions via low-rank embeddings, improving on linear models while staying cheaper than GBDT/MLP. We treat each of the 14 input bytes as a categorical field with 256 values (ordinal bins already quantized). A binary FM with K=8 latent factors adds ~36k parameters, trained with Adam on log-loss. Expect dev accuracy ≥0.860 (beating label-linear's 0.8524) at ~1500 Ir/item (between label-linear and label-mlp16). Failure: accuracy ≤0.852 or cost >3000 Ir/item.
```
```python
"""Factorization machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encodes each of the d input bytes as a categorical field (256 values).
Model: bias + sum_j w_j[x_j] + 1/2 sum_k ( (sum_j v_jk[x_j])^2 - sum_j v_jk[x_j]^2 )
Output: class 1 if logit>0 else 0 (equivalent to 2-class softmax with class 0 as reference).
Weights file format (little-endian):
  uint32 magic=0x464D4231 ('FMB1'), int32 d, int32 K, int32 C(=2)
  float32 bias
  float32 linear_weights[d][256]
  float32 embeddings[d][256][K]  (row-major: d*256*K)
Hyperparameters chosen on last 20% of train.in (file order), then retrain on all.
"""
import json, math, struct, sys

import numpy as np

# Grid search space (tuned on 20% holdout)
GRID = {
    'K': [4, 8],
    'lr': [3e-3, 1e-2],
    'L2': [1e-4, 1e-3],
}
SEED = 1
BATCH = 64
EPOCHS_GRID = 20
EPOCHS_FULL = 100
SAMPLES = 600000  # max samples per epoch (like label-linear)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, K, rng):
    linear = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    embed = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)
    return linear, embed, bias


def compute_logits(X, linear, embed, bias):
    """X: (n, d) uint8. Returns logits (n,) float32."""
    n, d = X.shape
    K = embed.shape[2]
    # linear part
    lin = bias + np.sum(linear[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part
    emb = embed[np.arange(d), X.T]  # (d, n, K) -> transpose to (n, d, K)
    emb = np.transpose(emb, (1, 0, 2))
    sum_emb = np.sum(emb, axis=1)  # (n, K)
    sum_emb_sq = np.sum(emb * emb, axis=1)
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    return (lin + inter).astype(np.float32)


def accuracy(logits, y):
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)


def train_model(X_tr, y_tr, X_val, y_val, K, lr, L2, epochs, rng):
    d = X_tr.shape[1]
    linear, embed, bias = init_params(d, K, rng)
    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    m_bias, v_bias = 0.0, 0.0
    t = 0
    n = X_tr.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            n_b = Xb.shape[0]
            emb_b = embed[np.arange(d), Xb.T]  # (d, n_b, K)
            emb_b = np.transpose(emb_b, (1, 0, 2))  # (n_b, d, K)
            sum_emb = np.sum(emb_b, axis=1)  # (n_b, K)
            sum_emb_sq = np.sum(emb_b * emb_b, axis=1)
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            lin = bias + np.sum(linear[np.arange(d), Xb.T], axis=0)
            logits = lin + inter
            # gradient of log-loss: p - y, where p = sigmoid(logits)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yb  # (n_b,)
            # gradients
            g_bias = np.mean(grad)
            # linear weights gradient
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(g_lin[j], vals, grad)
            g_lin /= n_b
            # embeddings gradient
            g_emb = np.zeros_like(embed)
            # d(inter)/d v_jk = grad * (sum_emb_k - v_jk)
            for j in range(d):
                vals = Xb[:, j]
                for k in range(K):
                    # sum_emb_k per sample
                    sum_emb_k = sum_emb[:, k]
                    v_jk = embed[j, vals, k]
                    g = grad * (sum_emb_k - v_jk)
                    np.add.at(g_emb[j, :, k], vals, g)
            g_emb /= n_b
            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * embed
            g_bias += L2 * bias
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
            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            embed -= lr * m_hat / (np.sqrt(v_hat) + eps)
    # validation accuracy
    logits_val = compute_logits(X_val, linear, embed, bias)
    return accuracy(logits_val, y_val), linear, embed, bias


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_state = None

    for K in GRID['K']:
        for lr in GRID['lr']:
            for L2 in GRID['L2']:
                rng_grid = np.random.default_rng(SEED)  # same init for each config
                acc, linear, embed, bias = train_model(X_tr, y_tr, X_val, y_val,
                                                       K, lr, L2, EPOCHS_GRID, rng_grid)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, L2)
                    best_state = (linear.copy(), embed.copy(), bias)

    print(f"Best grid: K={best_params[0]}, lr={best_params[1]}, L2={best_params[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best params
    K, lr, L2 = best_params
    rng_full = np.random.default_rng(SEED)
    _, linear, embed, bias = train_model(X, y, X, y, K, lr, L2, EPOCHS_FULL, rng_full)

    # Write weights.bin
    magic = 0x464D4231  # 'FMB1'
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1).
 * Weights format (little-endian):
 *   uint32 magic=0x464D4231, int32 d, int32 K, int32 C(=2)
 *   float32 bias
 *   float32 linear[d][256]
 *   float32 embed[d][256][K] (row-major)
 */
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
    if (nw < 16 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = *p++;
    const float *linear = p; p += (long)d * 256;
    const float *embed = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[32], sum_emb_sq[32];  // K <= 32 (we use K=8 max)

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        for (int kk = 0; kk < K; kk++) {
            sum_emb[kk] = 0.0f;
            sum_emb_sq[kk] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = embed + ((long)j * 256 + v) * K;
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
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```