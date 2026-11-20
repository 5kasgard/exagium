```hypothesis
Family: label-rff (Random Fourier Features + linear classifier).
Why: Optdigits pixels have local correlations; RBF kernel works well but is slow. RFF approximates RBF with a small explicit feature map, enabling fast linear inference. With D=32-256 features, it should beat linear (0.961) and approach MLP (0.977) at moderate cost.
Failure: If validation accuracy does not exceed label-linear by >0.5% or inference cost exceeds label-mlp16.
```

```python
"""label-rff: Random Fourier Features + linear classifier for optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_D = [16, 32, 64, 128, 256]  # RFF dimensions to try on tuning split
GAMMA = 1.0  # RBF kernel bandwidth parameter
MAGIC = 0x52464631  # 'RFF1' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_logistic(Z, y, C, rng, epochs, batch, lr, l2):
    """Train softmax classifier with Adam. Returns W (D,C), b (C)."""
    n, D = Z.shape
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # forward
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            # gradients
            gW = z.T @ grad_logits + l2 * W
            gb = grad_logits.sum(axis=0)
            grads = [gW, gb]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params[0], params[1]


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    # Base RNG for reproducibility
    base_rng = np.random.default_rng(SEED)

    # Compute normalization on training split
    X_train = X[idx_train].astype(np.float64)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    def normalize(X_sub):
        return (X_sub.astype(np.float64) - mean) / std

    X_train_norm = normalize(X[idx_train])
    X_tune_norm = normalize(X[idx_tune])
    y_train = y[idx_train]
    y_tune = y[idx_tune]

    best_D = None
    best_acc = -1.0
    best_W = best_b = None

    for D in CANDIDATE_D:
        # Generate RFF parameters deterministically per D
        rff_rng = np.random.default_rng(SEED + D * 10007)
        W_rff = (rff_rng.normal(0, GAMMA, (D, d))).astype(np.float32)
        b_rff = (rff_rng.uniform(0, 2 * math.pi, D)).astype(np.float32)
        scale = math.sqrt(2.0 / D)

        # Transform
        Z_train = (np.cos(X_train_norm @ W_rff.T + b_rff) * scale).astype(np.float32)
        Z_tune = (np.cos(X_tune_norm @ W_rff.T + b_rff) * scale).astype(np.float32)

        # Train logistic regression
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_train)))
        train_rng = np.random.default_rng(SEED + D * 10009)
        W_out, b_out = train_logistic(Z_train, y_train, C, train_rng, epochs, BATCH, LR, L2)

        # Evaluate
        logits = Z_tune @ W_out + b_out
        preds = logits.argmax(axis=1)
        acc = (preds == y_tune).mean()
        print(f"D={D}: tuning accuracy={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_W, best_b = W_out.copy(), b_out.copy()
            best_W_rff, best_b_rff = W_rff.copy(), b_rff.copy()

    print(f"Selected D={best_D} with acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best_D
    # Recompute normalization on all data
    mean_all = X.astype(np.float64).mean(axis=0)
    std_all = X.astype(np.float64).std(axis=0)
    std_all[std_all == 0] = 1.0
    X_all_norm = (X.astype(np.float64) - mean_all) / std_all

    # Regenerate RFF for best_D (same seed)
    rff_rng = np.random.default_rng(SEED + best_D * 10007)
    W_rff = (rff_rng.normal(0, GAMMA, (best_D, d))).astype(np.float32)
    b_rff = (rff_rng.uniform(0, 2 * math.pi, best_D)).astype(np.float32)
    scale = math.sqrt(2.0 / best_D)

    Z_all = (np.cos(X_all_norm @ W_rff.T + b_rff) * scale).astype(np.float32)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_rng = np.random.default_rng(SEED + best_D * 10009)
    W_out, b_out = train_logistic(Z_all, y, C, train_rng, epochs, BATCH, LR, L2)

    # Pack weights file
    # Order: magic, d, D, C, gamma, mean[d], std[d], W_rff[D][d], b_rff[D], W_out[D][C], b_out[C]
    header = struct.pack("<4if", MAGIC, d, best_D, C, GAMMA)
    mean_arr = mean_all.astype(np.float32)
    std_arr = std_all.astype(np.float32)
    out = header
    out += mean_arr.tobytes()
    out += std_arr.tobytes()
    out += W_rff.astype(np.float32).tobytes()
    out += b_rff.astype(np.float32).tobytes()
    out += W_out.astype(np.float32).tobytes()
    out += b_out.astype(np.float32).tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (pixels 0..16). Output: one byte class 0..9.
 * Weights format: magic(0x52464631), d, D, C, gamma, mean[d], std[d], W_rff[D][d], b_rff[D], W_out[D][C], b_out[C] (all float32). */
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
    if (nw < 20 || hdr[0] != 0x52464631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], D = hdr[2], C = hdr[3];
    float gamma = *(float *)(hdr + 4);
    if (d != 64 || C != 10 || D <= 0 || D > 1024) { fprintf(stderr, "bad header dims\n"); return 3; }

    float *ptr = (float *)(hdr + 5);  // after magic,d,D,C,gamma
    float *mean = ptr; ptr += d;
    float *std = ptr; ptr += d;
    float *W_rff = ptr; ptr += (long)D * d;
    float *b_rff = ptr; ptr += D;
    float *W_out = ptr; ptr += (long)D * C;
    float *b_out = ptr; ptr += C;

    long expected = 20 + (long)d * 4 + (long)d * 4 + (long)D * d * 4 + (long)D * 4 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_norm = malloc(d * sizeof(float));
    float *hidden = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));
    float scale = sqrtf(2.0f / D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize
        for (int j = 0; j < d; j++) {
            float v = (float)xi[j];
            x_norm[j] = (v - mean[j]) / std[j];
        }

        // RFF projection: hidden = cos(W_rff * x_norm + b_rff) * scale
        for (int u = 0; u < D; u++) {
            float sum = b_rff[u];
            const float *wrow = W_rff + (long)u * d;
            for (int j = 0; j < d; j++) sum += x_norm[j] * wrow[j];
            hidden[u] = cosf(sum) * scale;
        }

        // Linear classifier
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int u = 0; u < D; u++) {
            float h = hidden[u];
            const float *wrow = W_out + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += h * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```