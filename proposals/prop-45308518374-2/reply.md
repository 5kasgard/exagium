```hypothesis
Propose label-residual: a wide-and-deep model that adds a linear (wide) component to a one-hidden-layer MLP (deep) via a skip connection. The linear part captures strong first-order byte-position statistics; the hidden layer captures pairwise interactions. Hyperparameters (hidden size H, learning rate, L2) are tuned on the last 20% of train.in, then the model is retrained on all data. This should beat label-linear because it retains the linear baseline's strength while adding non-linearity, and tuning prevents overfitting. If dev accuracy does not exceed label-linear's 0.3375, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed seeds for determinism
SEED_BASE = 1
# Grid search space
H_CHOICES = [16, 32, 64]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-5]
BATCH = 64
MAX_EPOCHS_TUNE = 15
MAX_EPOCHS_FULL = 50
SAMPLES_PER_EPOCH = 600000  # used to cap epochs

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def relu(x):
    return np.maximum(x, 0, dtype=np.float32)

def softmax_cross_entropy_grad(logits, y):
    # logits: (batch, C), y: (batch,) int64
    batch = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch

def train_model(X, y, H, lr, l2, max_epochs, seed):
    n, d = X.shape
    C = 65
    rng = np.random.default_rng(seed)
    # Initialize parameters
    linear_emb = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 * C))).astype(np.float32)
    hidden_emb = (rng.uniform(-1, 1, (d, 256, H)) * math.sqrt(6.0 / (256 * H))).astype(np.float32)
    hidden_bias = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    output_bias = np.zeros(C, np.float32)
    params = [linear_emb, hidden_emb, hidden_bias, W2, output_bias]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]  # (batch, d)
            yb = y[idx]
            # Forward
            batch = xb.shape[0]
            # linear logits
            lin_logits = output_bias.copy()
            for j in range(d):
                lin_logits += linear_emb[j, xb[:, j], :]
            # hidden activations
            hid = hidden_bias.copy()
            for j in range(d):
                hid += hidden_emb[j, xb[:, j], :]
            hid = relu(hid)
            logits = lin_logits + hid @ W2
            # Gradient
            g = softmax_cross_entropy_grad(logits, yb)  # (batch, C)
            # Gradients for output_bias and linear_emb
            g_lin_bias = g.sum(axis=0)
            g_lin_emb = np.zeros_like(linear_emb)
            for j in range(d):
                # For each position, scatter-add g into g_lin_emb[j, byte, :]
                np.add.at(g_lin_emb[j], xb[:, j], g)
            # Gradients for W2 and hidden layer
            g_W2 = hid.T @ g
            g_hid = g @ W2.T * (hid > 0)
            g_hid_bias = g_hid.sum(axis=0)
            g_hid_emb = np.zeros_like(hidden_emb)
            for j in range(d):
                np.add.at(g_hid_emb[j], xb[:, j], g_hid)
            # L2 regularization
            g_lin_emb += l2 * linear_emb
            g_hid_emb += l2 * hidden_emb
            g_W2 += l2 * W2
            # Adam update
            grads = [g_lin_emb, g_hid_emb, g_hid_bias, g_W2, g_lin_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(X, y, params):
    linear_emb, hidden_emb, hidden_bias, W2, output_bias = params
    n, d = X.shape
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        batch = xb.shape[0]
        lin_logits = np.tile(output_bias, (batch, 1))
        for j in range(d):
            lin_logits += linear_emb[j, xb[:, j], :]
        hid = np.tile(hidden_bias, (batch, 1))
        for j in range(d):
            hid += hidden_emb[j, xb[:, j], :]
        hid = relu(hid)
        logits = lin_logits + hid @ W2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X_all.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n_total)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    best_acc = -1.0
    best_params = None
    best_hyper = None

    # Grid search
    for H in H_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # Deterministic seed per hyperparameter combo
                seed = SEED_BASE + H * 1000 + int(lr * 1e5) + int(l2 * 1e7)
                params = train_model(X_train, y_train, H, lr, l2, MAX_EPOCHS_TUNE, seed)
                acc = evaluate(X_tune, y_tune, params)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_hyper = (H, lr, l2)
                print(f"H={H} lr={lr} l2={l2} tune_acc={acc:.4f}", file=sys.stderr)

    H_best, lr_best, l2_best = best_hyper
    print(f"Best: H={H_best} lr={lr_best} l2={l2_best} tune_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    final_params = train_model(X_all, y_all, H_best, lr_best, l2_best, MAX_EPOCHS_FULL, SEED_BASE + 9999)

    # Export weights
    linear_emb, hidden_emb, hidden_bias, W2, output_bias = final_params
    d = X_all.shape[1]
    C = 65
    H = H_best
    magic = 0x52455344  # "RESD" for residual
    # Layout: magic, d, H, C; then linear_emb (d*256*C), hidden_emb (d*256*H),
    # hidden_bias (H), W2 (H*C), output_bias (C)
    header = struct.pack("<4i", magic, d, H, C)
    out = header
    out += linear_emb.astype("<f4").tobytes()
    out += hidden_emb.astype("<f4").tobytes()
    out += hidden_bias.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += output_bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Residual MLP (wide + deep) for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index.
 * Weights format: magic 0x52455344, int32 d, H, C;
 * float32 linear_emb[d][256][C], hidden_emb[d][256][H], hidden_bias[H], W2[H][C], output_bias[C]. */
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
    if (nw < 16 || hd[0] != 0x52455344) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16
        + 4L * d * 256 * C
        + 4L * d * 256 * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    const float *linear_emb = p; p += d * 256 * C;
    const float *hidden_emb = p; p += d * 256 * H;
    const float *hidden_bias = p; p += H;
    const float *W2 = p; p += H * C;
    const float *output_bias = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_logits = malloc(sizeof(float) * C);
    float *hid = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // linear logits = output_bias
        memcpy(lin_logits, output_bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *emb = linear_emb + (j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin_logits[c] += emb[c];
        }
        // hidden activations = hidden_bias
        memcpy(hid, hidden_bias, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *emb = hidden_emb + (j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hid[h] += emb[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hid[h] < 0.0f) hid[h] = 0.0f;
        // logits = lin_logits + hid @ W2
        memcpy(logits, lin_logits, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hid[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + h * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```