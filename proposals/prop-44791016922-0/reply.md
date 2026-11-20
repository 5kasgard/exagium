```hypothesis
label-fm: Factorization Machine with low-rank pairwise interactions (k=8). Adult income has strong feature interactions (e.g., education×occupation, marital×sex) that linear models miss. FM adds only O(d·k) parameters and computes interactions in O(d·k) time, keeping inference fast. If dev accuracy does not exceed label-linear by ≥0.5% (i.e., <0.857), the extra complexity is not justified.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning split (last 20% of train.in)
GRID = {
    'k': [4, 8, 12],
    'lr': [0.01, 0.03],
    'l2': [1e-4, 1e-3],
    'epochs': 20,      # epochs per config on tuning split
    'batch': 256,
    'seed': 1,
}
# After selection, retrain on full data with these settings
RETRAIN_EPOCHS = 30
RETRAIN_BATCH = 256

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X, params, k):
    """X: (B, d) uint8, params: (bias, W_lin, V)"""
    bias, W_lin, V = params
    B, d = X.shape
    # linear part
    lin = bias + np.sum(W_lin[np.arange(d), X.T], axis=0)  # (B,)
    # interaction part
    # gather embeddings: for each feature j, get (B, k)
    emb_sum = np.zeros((B, k), dtype=np.float32)
    emb_sum2 = np.zeros((B, k), dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j], :]          # (B, k)
        emb_sum += v
        emb_sum2 += v * v
    inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sum2, axis=1)
    return lin + inter

def fm_backward(X, y, params, k, l2):
    """Compute gradients for a minibatch. Returns (grad_bias, grad_W, grad_V)."""
    bias, W_lin, V = params
    B, d = X.shape
    # forward pass to get scores and probabilities
    lin = bias + np.sum(W_lin[np.arange(d), X.T], axis=0)
    emb_sum = np.zeros((B, k), dtype=np.float32)
    for j in range(d):
        emb_sum += V[j, X[:, j], :]
    emb_sum2 = np.zeros((B, k), dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j], :]
        emb_sum2 += v * v
    inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sum2, axis=1)
    scores = lin + inter
    probs = sigmoid(scores)
    dL_ds = (probs - y).astype(np.float32)   # (B,)

    grad_bias = np.sum(dL_ds)
    grad_W = np.zeros_like(W_lin)
    grad_V = np.zeros_like(V)

    # gradient for linear weights
    for j in range(d):
        np.add.at(grad_W[j], X[:, j], dL_ds)

    # gradient for embeddings
    for j in range(d):
        v = V[j, X[:, j], :]                 # (B, k)
        # dL/dV = dL_ds[:, None] * (emb_sum - v)
        g = dL_ds[:, None] * (emb_sum - v)   # (B, k)
        np.add.at(grad_V[j], X[:, j], g)

    # L2 regularization
    grad_W += l2 * W_lin
    grad_V += l2 * V
    # bias not regularized
    return grad_bias, grad_W, grad_V

def train_fm(X, y, k, lr, l2, epochs, batch, seed, verbose=False):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # init parameters
    bias = np.float32(0.0)
    W_lin = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    params = [bias, W_lin, V]
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            g_bias, g_W, g_V = fm_backward(xb, yb, params, k, l2)
            t += 1
            # update bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # update W_lin
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W_lin -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # update V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
            params = [bias, W_lin, V]
        if verbose:
            scores = fm_forward(X, params, k)
            preds = (scores > 0).astype(np.int64)
            acc = np.mean(preds == y)
            print(f"  epoch {epoch+1}: train acc={acc:.4f}")
    return params

def evaluate(X, y, params, k):
    scores = fm_forward(X, params, k)
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    val_size = n // 5
    # tuning split: last 20% in file order
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                cfg = (k, lr, l2)
                params = train_fm(X_train, y_train, k, lr, l2, GRID['epochs'], GRID['batch'], GRID['seed'])
                acc = evaluate(X_val, y_val, params, k)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = cfg
                    best_params = params
    print(f"Best config: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}")

    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    final_params = train_fm(X, y, k, lr, l2, RETRAIN_EPOCHS, RETRAIN_BATCH, GRID['seed'])

    # Save weights
    bias, W_lin, V = final_params
    magic = 0x464D4C31  # 'FML1'
    C = card["classes"]
    header = struct.pack("<4i", magic, d, k, C)
    # bias (float32), then W_lin (d*256), then V (d*256*k)
    out = header + np.array([bias], dtype=np.float32).tobytes() + W_lin.astype("<f4").tobytes() + V.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k < 1 || k > 32 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float lin = bias;
        for (int j = 0; j < d; j++) {
            lin += W_lin[j * 256 + xi[j]];
        }

        float sum[32] = {0}, sum2[32] = {0};
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                sum[f] += val;
                sum2[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) {
            inter += sum[f] * sum[f] - sum2[f];
        }
        inter *= 0.5f;
        float score = lin + inter;
        pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```