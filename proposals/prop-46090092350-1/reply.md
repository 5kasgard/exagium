```hypothesis
Family: label-fm (factorization machine). Optdigits pixels have strong pairwise interactions (strokes); a degree-2 polynomial with low-rank factors captures these non-linearities better than a linear model while adding only O(C*d*k) parameters. With k=8 the model has ~5k parameters and ~6k FLOPs/item, far cheaper than GBDT/MLP. If dev accuracy does not exceed label-linear (0.9613), the low-rank interaction hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed seed for full determinism
SEED = 1
# Grid search space (tuned on last 20% of train.in)
K_CHOICES = [4, 8, 12]
LR_CHOICES = [0.01, 0.003]
L2_CHOICES = [1e-4, 1e-3]
EPOCHS = 30
BATCH = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(scores, y):
    # scores: (batch, C), y: (batch,)
    batch, C = scores.shape
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch  # (batch, C)

def forward(Z, W_lin, b, V, norm_V_sq):
    # Z: (batch, d)
    batch, d = Z.shape
    C, _, k = V.shape
    linear = Z @ W_lin.T + b  # (batch, C)
    interaction = np.zeros((batch, C), dtype=np.float32)
    for c in range(C):
        A = Z @ V[c]                 # (batch, k)
        term1 = np.sum(A * A, axis=1)
        term2 = np.sum(Z * Z * norm_V_sq[c], axis=1)
        interaction[:, c] = 0.5 * (term1 - term2)
    return linear + interaction

def train_model(Z_train, y_train, Z_val, y_val, C, k, lr, l2, rng):
    n, d = Z_train.shape
    # Initialize parameters
    W_lin = rng.normal(0, 0.01, (C, d)).astype(np.float32)
    b = np.zeros(C, np.float32)
    V = rng.normal(0, 0.01, (C, d, k)).astype(np.float32)
    # Adam state
    m_W, v_W = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    best_val_acc = -1.0
    best_params = None

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Z = Z_train[idx]
            y = y_train[idx]
            batch = len(y)

            # Forward
            norm_V_sq = np.sum(V * V, axis=2)  # (C, d)
            scores = forward(Z, W_lin, b, V, norm_V_sq)
            g = softmax_cross_entropy_grad(scores, y)  # (batch, C)

            # Gradients
            # Linear
            grad_W_lin = Z.T @ g + l2 * W_lin  # (d, C) -> transpose later
            grad_b = g.sum(axis=0) + l2 * b

            # V gradients
            grad_V = np.zeros_like(V)
            Z_sq = Z * Z
            for c in range(C):
                A = Z @ V[c]                     # (batch, k)
                g_c = g[:, c:c+1]                # (batch, 1)
                # term1 gradient: Z.T @ (g_c * A)
                grad_V[c] += Z.T @ (g_c * A)
                # term2 gradient: - (Z^2).T @ (g_c * V[c])
                grad_V[c] -= Z_sq.T @ (g_c * V[c])
                grad_V[c] += l2 * V[c]

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # W_lin
            m_W = beta1 * m_W + (1 - beta1) * grad_W_lin.T
            v_W = beta2 * v_W + (1 - beta2) * (grad_W_lin.T ** 2)
            W_lin -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            b -= lr * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

        # Validation accuracy
        norm_V_sq = np.sum(V * V, axis=2)
        val_scores = forward(Z_val, W_lin, b, V, norm_V_sq)
        val_pred = np.argmax(val_scores, axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (W_lin.copy(), b.copy(), V.copy())

    return best_params, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Normalize features (ordinal -> z-score)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)

    # Split: last 20% for tuning
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                params, val_acc = train_model(Z_train, y_train, Z_tune, y_tune, C, k, lr, l2, rng)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cfg = (k, lr, l2)
                    best_params = params

    # Retrain on full train with best config
    k, lr, l2 = best_cfg
    W_lin, b, V = best_params
    # Re-initialize Adam state and continue training on full data for EPOCHS more epochs?
    # Simpler: just train from scratch on full data with same epochs (deterministic because rng state advanced).
    # But we must ensure the final weights are exactly reproducible. We'll re-run training on full data
    # with the same hyperparameters and a fresh rng? However, the rng state has advanced through grid search.
    # To guarantee byte-identical weights across runs, we must make the full training deterministic
    # independent of grid search randomness. Easiest: reset rng to a fixed seed derived from SEED and best_cfg,
    # then train on full data. But the contract says "Running it twice must give byte-identical weights.bin".
    # Since the grid search is part of train.py, the entire sequence must be deterministic.
    # Our current rng is global and advances through grid search. That means the full training will use
    # the rng state after grid search, which is deterministic because grid search order is fixed.
    # So we can just continue using the same rng.
    norm_V_sq = np.sum(V * V, axis=2)
    # We'll do a few more epochs on full data? Actually we already trained on train_idx for EPOCHS.
    # We can just train on full data for EPOCHS epochs starting from the best_params.
    # But we must update the parameters with the same optimizer state? To keep it simple, we'll
    # re-initialize optimizer state and train on full data for EPOCHS epochs.
    # However, the rng state will be different from a fresh run because we consumed random numbers
    # during grid search. That's fine as long as we don't reset the seed.
    # We'll just continue training on full data for EPOCHS epochs.
    m_W, v_W = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Z_batch = Z[idx]
            y_batch = y[idx]
            batch = len(y_batch)

            norm_V_sq = np.sum(V * V, axis=2)
            scores = forward(Z_batch, W_lin, b, V, norm_V_sq)
            g = softmax_cross_entropy_grad(scores, y_batch)

            grad_W_lin = Z_batch.T @ g + l2 * W_lin
            grad_b = g.sum(axis=0) + l2 * b

            grad_V = np.zeros_like(V)
            Z_sq = Z_batch * Z_batch
            for c in range(C):
                A = Z_batch @ V[c]
                g_c = g[:, c:c+1]
                grad_V[c] += Z_batch.T @ (g_c * A)
                grad_V[c] -= Z_sq.T @ (g_c * V[c])
                grad_V[c] += l2 * V[c]

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_W = beta1 * m_W + (1 - beta1) * grad_W_lin.T
            v_W = beta2 * v_W + (1 - beta2) * (grad_W_lin.T ** 2)
            W_lin -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            b -= lr * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

    # Write weights.bin
    # Format: magic, d, C, k, mean[d], std[d], W_lin[C*d], b[C], V[C*d*k] (all float32 little-endian)
    magic = 0x464D5F31  # "FM_1"
    k = best_cfg[0]
    out = struct.pack("<4i", magic, d, C, k)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 16 || hdr[0] != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], k = hdr[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)C * d + C + (long)C * d * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *mean = (float *)(hdr + 4);
    float *std = mean + d;
    float *W_lin = std + d;
    float *b = W_lin + (long)C * d;
    float *V = b + C;

    // Precompute norm_V_sq[C][d]
    float *norm_V_sq = malloc((long)C * d * sizeof(float));
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int f = 0; f < k; f++) {
                float v = V[((long)c * d + i) * k + f];
                sum += v * v;
            }
            norm_V_sq[(long)c * d + i] = sum;
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));
    float *A = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        // Normalize
        for (int j = 0; j < d; j++) {
            float val = xi[j];
            float s = std[j];
            xnorm[j] = (s != 0.0f) ? (val - mean[j]) / s : 0.0f;
        }
        // Compute scores
        for (int c = 0; c < C; c++) {
            // Linear part
            float lin = b[c];
            for (int j = 0; j < d; j++) lin += xnorm[j] * W_lin[(long)c * d + j];
            // Interaction
            // A_f = sum_j V[c,j,f] * xnorm[j]
            for (int f = 0; f < k; f++) A[f] = 0.0f;
            for (int j = 0; j < d; j++) {
                float xj = xnorm[j];
                if (xj == 0.0f) continue;
                for (int f = 0; f < k; f++) {
                    A[f] += V[((long)c * d + j) * k + f] * xj;
                }
            }
            float term1 = 0.0f;
            for (int f = 0; f < k; f++) term1 += A[f] * A[f];
            float term2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float xj = xnorm[j];
                term2 += xj * xj * norm_V_sq[(long)c * d + j];
            }
            float inter = 0.5f * (term1 - term2);
            scores[c] = lin + inter;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```