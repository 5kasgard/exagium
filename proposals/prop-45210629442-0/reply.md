```hypothesis
Family: label-gam (Generalized Additive Model). Learns a per-feature per-class lookup table (14×256×2) so each feature can contribute an arbitrary non-linear score, then sums them for a linear decision. Adult's ordinal features (age, hours/week, capital gain/loss) have strongly non-linear relationships with income; a GAM captures this without feature interactions. Inference cost identical to label-linear (14 table lookups + 14×2 adds). If dev accuracy does not exceed label-linear (0.8524), it means either the non-linearities are not helpful or the extra capacity overfits despite L2/early stopping.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning on the last 20% of train.in
GRID = {
    'lr': [3e-3, 1e-3],
    'l2': [1e-3, 1e-4],
    'epochs': [50],
    'batch': [64]
}
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, lr, l2, epochs, batch, rng):
    n, d = X.shape
    # Parameters: T (d*256, C), bias (C)
    T = rng.normal(0, 0.01, (d * 256, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    # Adam state
    m_T = np.zeros_like(T); v_T = np.zeros_like(T)
    m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            x = X[idx]
            yy = y[idx]
            bsize = len(yy)
            # Forward
            scores = np.zeros((bsize, C), np.float32)
            for j in range(d):
                scores += T[j*256 + x[:, j], :]
            scores += bias
            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            # Loss gradient
            dscores = probs
            dscores[np.arange(bsize), yy] -= 1.0
            dscores /= bsize
            # Backward
            dbias = dscores.sum(axis=0)
            indices_flat = (np.arange(d) * 256 + x.T).T.flatten()
            dscores_rep = np.repeat(dscores, d, axis=0)
            dT = np.zeros_like(T)
            np.add.at(dT, indices_flat, dscores_rep)
            dT += l2 * T
            dbias += l2 * bias
            # Adam update
            t += 1
            m_T = beta1 * m_T + (1 - beta1) * dT
            v_T = beta2 * v_T + (1 - beta2) * dT * dT
            m_hat_T = m_T / (1 - beta1 ** t)
            v_hat_T = v_T / (1 - beta2 ** t)
            T -= lr * m_hat_T / (np.sqrt(v_hat_T) + eps)
            m_b = beta1 * m_b + (1 - beta1) * dbias
            v_b = beta2 * v_b + (1 - beta2) * dbias * dbias
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
    return T, bias

def evaluate(X, y, T, bias, C):
    n, d = X.shape
    scores = np.zeros((n, C), np.float32)
    for j in range(d):
        scores += T[j*256 + X[:, j], :]
    scores += bias
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X)
    # Split: last 20% for tuning (file order)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_T = None
    best_bias = None
    
    # Grid search on tuning split
    for lr in GRID['lr']:
        for l2 in GRID['l2']:
            for epochs in GRID['epochs']:
                for batch in GRID['batch']:
                    config_rng = np.random.default_rng(SEED + hash((lr, l2, epochs, batch)) % 2**32)
                    T, bias = train_model(X_train, y_train, C, lr, l2, epochs, batch, config_rng)
                    acc = evaluate(X_tune, y_tune, T, bias, C)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (lr, l2, epochs, batch)
                        best_T = T.copy()
                        best_bias = bias.copy()
    # Retrain on all data with best hyperparameters
    final_rng = np.random.default_rng(SEED + 12345)
    T_final, bias_final = train_model(X, y, C, *best_params, final_rng)
    
    # Write weights in label-linear format: magic, d, H=0, C, T (d*256*C floats), bias (C floats)
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T_final.astype("<f4").tobytes()
    out += bias_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```