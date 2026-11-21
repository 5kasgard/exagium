```hypothesis
We propose a "label-residual" family: a linear model (per-position per-byte lookup tables) plus a small ReLU MLP with a bottleneck hidden layer (size H). The linear part captures first-order statistics; the MLP captures pairwise interactions via a low-rank factorization. H is chosen on the tuning split (last 20% of train.in) then retrained on all data. This should beat label-linear (0.3375) because character prediction benefits from modeling interactions (e.g., "th"→"e"). If it does not improve, it suggests the interactions are not learnable with this capacity or the linear model is already optimal for 16-byte context.
```
```python
"""label-residual: linear + one-hidden-layer ReLU MLP with tuned H.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format (little-endian):
  magic(0x52455349), d, H, C (int32)
  T_lin[d*256*C] (float32), b_lin[C]
  T_nl[d*256*H] (float32), b_nl[H]
  W2[H*C] (float32), b2[C]"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Hidden sizes to try on validation split
H_CANDIDATES = [16, 32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, H, rng, epochs):
    """Train and return parameters for given H."""
    n = X.shape[0]
    # Initialize parameters
    # T_lin: (d, 256, C)
    T_lin = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    # T_nl: (d, 256, H)
    T_nl = (rng.uniform(-1, 1, (d, 256, H)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    b_nl = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [T_lin, b_lin, T_nl, b_nl, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            # Linear part
            lin = b_lin + np.sum(T_lin[np.arange(d), Xb.T], axis=0)  # (B, C)
            # Nonlinear part
            nl_pre = b_nl + np.sum(T_nl[np.arange(d), Xb.T], axis=0)  # (B, H)
            nl = np.maximum(nl_pre, 0)
            logits = lin + nl @ W2 + b2

            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B  # (B, C)

            # Gradients
            # Linear
            db_lin = np.sum(dlogits, axis=0)
            dT_lin = np.zeros_like(T_lin)
            for j in range(d):
                np.add.at(dT_lin[j], Xb[:, j], dlogits)

            # Nonlinear: backprop through ReLU and W2
            dnl = dlogits @ W2.T  # (B, H)
            dnl_pre = dnl * (nl_pre > 0)
            db_nl = np.sum(dnl_pre, axis=0)
            dT_nl = np.zeros_like(T_nl)
            for j in range(d):
                np.add.at(dT_nl[j], Xb[:, j], dnl_pre)

            dW2 = nl.T @ dlogits
            db2 = np.sum(dlogits, axis=0)

            grads = [dT_lin, db_lin, dT_nl, db_nl, dW2, db2]

            # L2 regularization on weights (not biases)
            grads[0] += L2 * T_lin
            grads[2] += L2 * T_nl
            grads[4] += L2 * W2

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params

def evaluate(X, y, params):
    T_lin, b_lin, T_nl, b_nl, W2, b2 = params
    lin = b_lin + np.sum(T_lin[np.arange(X.shape[1]), X.T], axis=0)
    nl_pre = b_nl + np.sum(T_nl[np.arange(X.shape[1]), X.T], axis=0)
    nl = np.maximum(nl_pre, 0)
    logits = lin + nl @ W2 + b2
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(y_all)
    # Split: last 20% for validation
    split = int(0.8 * n_all)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Try each H on validation split
    best_H, best_acc = H_CANDIDATES[0], -1.0
    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED + H)  # deterministic per H
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_model(X_train, y_train, d, C, H, rng, epochs)
        acc = evaluate(X_val, y_val, params)
        if acc > best_acc:
            best_acc, best_H = acc, H
        print(f"H={H} val_acc={acc:.4f}", file=sys.stderr)

    # Retrain on full data with best_H
    print(f"Selected H={best_H}, retraining on full data...", file=sys.stderr)
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    T_lin, b_lin, T_nl, b_nl, W2, b2 = train_model(X_all, y_all, d, C, best_H, rng, epochs)

    # Write weights.bin
    magic = 0x52455349  # "RESI"
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += T_nl.astype("<f4").tobytes()
    out += b_nl.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-residual infer: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weight format (little-endian):
 *   magic(0x52455349), d, H, C (int32)
 *   T_lin[d*256*C] (float32), b_lin[C]
 *   T_nl[d*256*H] (float32), b_nl[H]
 *   W2[H*C] (float32), b2[C] */
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
    if (nw < 16 || hd[0] != 0x52455349) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16
        + 4L * d * 256 * C + 4L * C
        + 4L * d * 256 * H + 4L * H
        + 4L * H * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *T_lin = p; p += (long)d * 256 * C;
    const float *b_lin = p; p += C;
    const float *T_nl  = p; p += (long)d * 256 * H;
    const float *b_nl  = p; p += H;
    const float *W2    = p; p += (long)H * C;
    const float *b2    = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *nl_acc  = malloc(sizeof(float) * H);
    float *scores  = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        memcpy(lin_acc, b_lin, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin_acc[c] += t[c];
        }

        // Nonlinear part
        memcpy(nl_acc, b_nl, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T_nl + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) nl_acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (nl_acc[h] < 0.0f) nl_acc[h] = 0.0f;

        // Combine
        memcpy(scores, b_lin, C * sizeof(float)); // start with linear bias? Wait: we already have lin_acc = b_lin + sum T_lin.
        // Actually scores should be lin_acc + nl_acc @ W2 + b2.
        // Let's compute properly:
        memcpy(scores, lin_acc, C * sizeof(float)); // scores = linear part
        for (int c = 0; c < C; c++) scores[c] += b2[c]; // add b2
        for (int h = 0; h < H; h++) {
            float nh = nl_acc[h];
            if (nh == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += nh * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```