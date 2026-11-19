```hypothesis
We propose a "label-cnn" family: a 1D convolutional neural network with shared byte embeddings, a single convolutional layer (kernel size 3) with global max pooling, and a linear output. This captures local character n-gram patterns (like trigrams) more effectively than a linear model, while remaining extremely fast at inference (only a few thousand FLOPs per item). We expect it to beat label-linear (0.3375) on dev accuracy, approaching label-gbdt (0.3923) with lower CPU cost. If it fails to exceed label-linear on the tuning split, the family does not work for this task.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed training hyperparameters (will be tuned on the last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning
GRID = [
    {"D": 8, "F": 32, "K": 3},
    {"D": 16, "F": 64, "K": 3},
    {"D": 8, "F": 64, "K": 3},
    {"D": 16, "F": 32, "K": 3},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(D, F, K, C, rng):
    # Embedding: 256 x D
    emb = (rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    # Conv: F x K x D
    conv_w = (rng.uniform(-1, 1, (F, K, D)) * math.sqrt(6.0 / (K * D))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    # Linear: F x C
    lin_w = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / (F + C))).astype(np.float32)
    lin_b = np.zeros(C, np.float32)
    return emb, conv_w, conv_b, lin_w, lin_b

def forward(emb, conv_w, conv_b, lin_w, lin_b, X):
    # X: (N, 16) uint8
    N = X.shape[0]
    # Embedding lookup: (N, 16, D)
    E = emb[X]  # (N, 16, D)
    # Convolution: for each filter, each position
    F, K, D = conv_w.shape
    P = 16 - K + 1
    # Compute conv output: (N, P, F)
    conv_out = np.zeros((N, P, F), np.float32)
    for f in range(F):
        for p in range(P):
            # dot product over K*D
            window = E[:, p:p+K, :]  # (N, K, D)
            w = conv_w[f]  # (K, D)
            conv_out[:, p, f] = np.sum(window * w, axis=(1,2)) + conv_b[f]
    # ReLU
    conv_out = np.maximum(conv_out, 0)
    # Global max pooling over positions: (N, F)
    pooled = np.max(conv_out, axis=1)
    # Linear: (N, C)
    logits = pooled @ lin_w + lin_b
    return logits, pooled, conv_out, E

def loss_and_grad(logits, y, pooled, conv_out, E, emb, conv_w, conv_b, lin_w, lin_b, X, L2):
    N, C = logits.shape
    # Softmax + cross-entropy gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N  # (N, C)
    # Linear gradients
    grad_lin_w = pooled.T @ probs + L2 * lin_w
    grad_lin_b = probs.sum(axis=0)
    # Backprop to pooled
    grad_pooled = probs @ lin_w.T  # (N, F)
    # Backprop through max pooling
    F = conv_w.shape[0]
    P = conv_out.shape[1]
    grad_conv_out = np.zeros_like(conv_out)
    # For each sample and filter, find argmax position
    for n in range(N):
        for f in range(F):
            p_max = np.argmax(conv_out[n, :, f])
            grad_conv_out[n, p_max, f] = grad_pooled[n, f]
    # Backprop through ReLU
    grad_conv_out *= (conv_out > 0)
    # Convolution gradients
    grad_conv_w = np.zeros_like(conv_w)
    grad_conv_b = np.zeros_like(conv_b)
    K, D = conv_w.shape[1], conv_w.shape[2]
    for f in range(F):
        for p in range(P):
            window = E[:, p:p+K, :]  # (N, K, D)
            g = grad_conv_out[:, p, f]  # (N,)
            # Gradient w.r.t conv_w[f] = sum_n g[n] * window[n]
            for k in range(K):
                for d in range(D):
                    grad_conv_w[f, k, d] = np.sum(g * window[:, k, d]) + L2 * conv_w[f, k, d]
        grad_conv_b[f] = np.sum(grad_conv_out[:, :, f])
    # Backprop to embedding
    grad_E = np.zeros_like(E)
    for f in range(F):
        for p in range(P):
            g = grad_conv_out[:, p, f][:, None, None]  # (N,1,1)
            w = conv_w[f][None, :, :]  # (1,K,D)
            grad_E[:, p:p+K, :] += g * w
    # Embedding gradients: accumulate over positions
    grad_emb = np.zeros_like(emb)
    np.add.at(grad_emb, X, grad_E)
    grad_emb += L2 * emb
    return grad_emb, grad_conv_w, grad_conv_b, grad_lin_w, grad_lin_b

def train_model(X, y, D, F, K, C, rng, tune_X=None, tune_y=None):
    params = init_params(D, F, K, C, rng)
    emb, conv_w, conv_b, lin_w, lin_b = params
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_acc = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits, pooled, conv_out, E = forward(emb, conv_w, conv_b, lin_w, lin_b, xb)
            grads = loss_and_grad(logits, yb, pooled, conv_out, E, emb, conv_w, conv_b, lin_w, lin_b, xb, L2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Evaluate on tuning set if provided
        if tune_X is not None:
            logits, _, _, _ = forward(emb, conv_w, conv_b, lin_w, lin_b, tune_X)
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == tune_y)
            if acc > best_acc:
                best_acc = acc
                best_params = [p.copy() for p in params]
    if best_params is not None:
        return best_params
    return params

def evaluate(X, y, params):
    emb, conv_w, conv_b, lin_w, lin_b = params
    logits, _, _, _ = forward(emb, conv_w, conv_b, lin_w, lin_b, X)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def pack_weights(d, D, F, K, C, params):
    emb, conv_w, conv_b, lin_w, lin_b = params
    magic = 0x314E4E43  # 'CNN1' little-endian
    header = struct.pack("<6i", magic, d, D, F, K, C)
    body = (emb.astype("<f4").tobytes() +
            conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            lin_w.astype("<f4").tobytes() +
            lin_b.astype("<f4").tobytes())
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    train_X, tune_X = X[:split], X[split:]
    train_y, tune_y = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_config = None
    best_acc = -1
    best_params = None
    for cfg in GRID:
        D, F, K = cfg["D"], cfg["F"], cfg["K"]
        # Train on train split
        params = train_model(train_X, train_y, D, F, K, C, rng, tune_X, tune_y)
        # Evaluate on tune split
        acc = evaluate(tune_X, tune_y, params)
        print(f"Config D={D} F={F} K={K}: tune acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_config = cfg
            best_params = params
    print(f"Best config: {best_config} with acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    D, F, K = best_config["D"], best_config["F"], best_config["K"]
    final_params = train_model(X, y, D, F, K, C, rng)
    # Write weights
    out = pack_weights(d, D, F, K, C, final_params)
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
    if (nw < 24 || hd[0] != 0x314E4E43) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], F = hd[3], K = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != 16 || K != 3) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 24
        + 4L * 256 * D
        + 4L * F * K * D
        + 4L * F
        + 4L * F * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *conv_w = emb + 256 * D;
    const float *conv_b = conv_w + F * K * D;
    const float *lin_w = conv_b + F;
    const float *lin_b = lin_w + F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Allocate buffers
    float *E = malloc(16 * D * sizeof(float));          // embeddings for one sample
    float *conv_out = malloc((16 - K + 1) * F * sizeof(float)); // (P, F)
    float *pooled = malloc(F * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int p = 0; p < d; p++) {
            const float *e = emb + xi[p] * D;
            memcpy(E + p * D, e, D * sizeof(float));
        }

        // Convolution + ReLU
        int P = d - K + 1;
        for (int f = 0; f < F; f++) {
            const float *w = conv_w + f * K * D;
            float b = conv_b[f];
            for (int p = 0; p < P; p++) {
                float sum = b;
                for (int kk = 0; kk < K; kk++) {
                    const float *e = E + (p + kk) * D;
                    for (int dd = 0; dd < D; dd++) {
                        sum += e[dd] * w[kk * D + dd];
                    }
                }
                conv_out[p * F + f] = sum > 0 ? sum : 0;
            }
        }

        // Global max pooling
        for (int f = 0; f < F; f++) {
            float maxv = -1e30f;
            for (int p = 0; p < P; p++) {
                float v = conv_out[p * F + f];
                if (v > maxv) maxv = v;
            }
            pooled[f] = maxv;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            for (int f = 0; f < F; f++) {
                sum += pooled[f] * lin_w[f * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(E); free(conv_out); free(pooled); free(logits);
    return 0;
}
```