```hypothesis
We propose a Factorization Machine (FM) of degree 2 with low-rank embeddings (k=4 or 8). FMs capture pairwise feature interactions efficiently, which are crucial for tabular data like Adult where combinations of categorical features (e.g., education+occupation) are predictive. The linear model cannot capture interactions; the MLP does but with more parameters. FMs are more parameter-efficient and should beat linear accuracy while remaining fast at inference. If dev accuracy does not exceed label-linear (0.8524), the FM fails to capture useful interactions.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning on the last 20% of train.in
GRID = [
    {"lr": 0.03, "epochs": 30, "k": 4, "l2": 1e-4, "batch": 64},
    {"lr": 0.01, "epochs": 30, "k": 4, "l2": 1e-4, "batch": 64},
    {"lr": 0.03, "epochs": 20, "k": 8, "l2": 1e-3, "batch": 64},
    {"lr": 0.01, "epochs": 20, "k": 8, "l2": 1e-3, "batch": 64},
]

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_fm(X, y, lr, epochs, k, l2, batch, rng):
    n, d = X.shape
    # Initialize weights
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.float32(0.0)
    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mb, vb = np.float32(0.0), np.float32(0.0)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)
            # Forward
            linear = bias + np.sum(W[np.arange(d), xb.T], axis=0)
            sum_v = np.zeros((B, k), dtype=np.float32)
            sum_sq = np.zeros((B, k), dtype=np.float32)
            for j in range(d):
                vj = V[j, xb[:, j]]
                sum_v += vj
                sum_sq += vj * vj
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
            logit = linear + interaction
            prob = sigmoid(logit)
            # Gradient
            grad = (prob - yb).astype(np.float32)
            # Backward
            g_bias = grad.mean()
            gW = np.zeros_like(W)
            for j in range(d):
                np.add.at(gW[j], xb[:, j], grad)
            gW /= B
            gW += l2 * W
            gV = np.zeros_like(V)
            for j in range(d):
                vj = V[j, xb[:, j]]
                sum_v_other = sum_v - vj
                grad_expanded = grad[:, None] * sum_v_other
                for f in range(k):
                    np.add.at(gV[j, :, f], xb[:, j], grad_expanded[:, f])
            gV /= B
            gV += l2 * V
            # Adam update
            t += 1
            mb = beta1 * mb + (1 - beta1) * g_bias
            vb = beta2 * vb + (1 - beta2) * g_bias * g_bias
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            bias -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV * gV
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
    return bias, W, V

def eval_fm(X, y, bias, W, V):
    n, d = X.shape
    k = V.shape[2]
    linear = bias + np.sum(W[np.arange(d), X.T], axis=0)
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_sq = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        vj = V[j, X[:, j]]
        sum_v += vj
        sum_sq += vj * vj
    interaction = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
    logit = linear + interaction
    prob = sigmoid(logit)
    pred = (prob > 0.5).astype(np.int64)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    # Tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    best_acc = -1
    best_params = None
    for params in GRID:
        bias, W, V = train_fm(X_train, y_train, params["lr"], params["epochs"], params["k"], params["l2"], params["batch"], rng)
        acc = eval_fm(X_tune, y_tune, bias, W, V)
        if acc > best_acc:
            best_acc = acc
            best_params = params
    # Retrain on full data with best hyperparameters
    bias, W, V = train_fm(X, y, best_params["lr"], best_params["epochs"], best_params["k"], best_params["l2"], best_params["batch"], rng)
    # Save weights
    k = best_params["k"]
    magic = 0x54464D31  # TFM1
    header = struct.pack("<4i", magic, d, k, C)
    bias_bytes = struct.pack("<f", bias)
    W_bytes = W.astype("<f4").tobytes()
    V_bytes = V.astype("<f4").tobytes()
    out = header + bias_bytes + W_bytes + V_bytes
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x54464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *ptr = (float *)(wb + 16);
    float bias = *ptr++;
    float *W = ptr; ptr += (long)d * 256;
    float *V = ptr;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = bias;
        for (int j = 0; j < d; j++) {
            linear += W[(long)j * 256 + xi[j]];
        }
        for (int f = 0; f < k; f++) {
            sum_v[f] = 0.0f;
            sum_sq[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float v = vj[f];
                sum_v[f] += v;
                sum_sq[f] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < k; f++) {
            interaction += sum_v[f] * sum_v[f] - sum_sq[f];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```