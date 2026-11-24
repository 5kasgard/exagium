```hypothesis
Family: label-fm (Factorization Machine with low-rank embeddings).
Why: Captures pairwise feature interactions (e.g., education×occupation) crucial for Adult income prediction, which linear models miss. Low-rank embeddings (K=8) keep inference fast (O(d*K) per item) and model size small (~100KB). Trained with Adam on cross-entropy, deterministic.
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear by >2×.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte value (0-255) per feature with a linear weight and a K-dim embedding.
Output: logits = w0_c + sum_j w_c[j][x_j] + 0.5*sum_k((sum_j v_c[j][x_j][k])^2 - sum_j v_c[j][x_j][k]^2)
Softmax over C classes. Hyperparameters chosen on last 20% of train.in (tuning split)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.003, 0.001]
L2_CHOICES = [0.001, 0.0001]
TUNE_EPOCHS = 10
FINAL_EPOCHS = 50
BATCH = 256


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, K, rng):
    w0 = np.zeros(C, np.float32)
    w = np.zeros((C, d, 256), np.float32)
    v = rng.normal(0, 1.0 / math.sqrt(K), (C, d, 256, K)).astype(np.float32)
    return w0, w, v


def forward(X, w0, w, v):
    """X: (B, d) uint8. Returns logits (B, C)."""
    B, d = X.shape
    C = w0.shape[0]
    K = v.shape[3]
    logits = np.broadcast_to(w0, (B, C)).copy()  # (B, C)
    for c in range(C):
        # linear term
        for j in range(d):
            logits[:, c] += w[c, j, X[:, j]]
        # interaction term
        sum_v = np.zeros((B, K), np.float32)
        sum_v_sq = np.zeros((B, K), np.float32)
        for j in range(d):
            v_vals = v[c, j, X[:, j], :]  # (B, K)
            sum_v += v_vals
            sum_v_sq += v_vals * v_vals
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)  # (B,)
        logits[:, c] += interaction
    return logits


def loss_and_grad(X, y, w0, w, v, L2):
    """Returns (loss, grads) where grads = (g_w0, g_w, g_v)."""
    B, d = X.shape
    C = w0.shape[0]
    K = v.shape[3]
    logits = forward(X, w0, w, v)
    # softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # cross-entropy loss
    loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
    # gradient w.r.t logits
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B  # (B, C)

    g_w0 = np.zeros_like(w0)
    g_w = np.zeros_like(w)
    g_v = np.zeros_like(v)

    for c in range(C):
        dc = dlogits[:, c]  # (B,)
        g_w0[c] = dc.sum()
        for j in range(d):
            vals = X[:, j]  # (B,)
            # gradient for w
            np.add.at(g_w[c, j], vals, dc)
            # gradient for v
            v_vals = v[c, j, vals, :]  # (B, K)
            sum_v = np.zeros((B, K), np.float32)
            for jj in range(d):
                sum_v += v[c, jj, X[:, jj], :]
            # d(interaction)/dv = sum_v - v_vals
            grad_v = dc[:, None] * (sum_v - v_vals)  # (B, K)
            np.add.at(g_v[c, j], vals, grad_v)
    # L2 regularization gradients
    g_w += L2 * w
    g_v += L2 * v
    return loss, (g_w0, g_w, g_v)


def adam_step(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params = []
    new_state = []
    for p, g, (m, v) in zip(params, grads, state):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_params.append(p)
        new_state.append((m, v))
    return new_params, new_state


def train_fm(X, y, d, C, K, lr, L2, epochs, batch, seed):
    rng = np.random.default_rng(seed)
    w0, w, v = init_params(d, C, K, rng)
    params = [w0, w, v]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            loss, grads = loss_and_grad(Xb, yb, w0, w, v, L2)
            t += 1
            params, state = adam_step(params, grads, state, t, lr)
            w0, w, v = params
    return w0, w, v


def accuracy(X, y, w0, w, v):
    logits = forward(X, w0, w, v)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_hp = None
    best_params = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                w0, w, v = train_fm(X_tr, y_tr, d, C, K, lr, L2, TUNE_EPOCHS, BATCH, SEED)
                acc = accuracy(X_tune, y_tune, w0, w, v)
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (K, lr, L2)
                    best_params = (w0, w, v)
                print(f"K={K} lr={lr} L2={L2} tune_acc={acc:.4f}", file=sys.stderr)

    K, lr, L2 = best_hp
    print(f"Best: K={K} lr={lr} L2={L2} tune_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    w0, w, v = train_fm(X, y, d, C, K, lr, L2, FINAL_EPOCHS, BATCH, SEED)

    # Write weights.bin
    # Format: magic, d, K, C; then w0[C], w[C][d][256], v[C][d][256][K] all float32 little-endian
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    data = bytearray()
    data += w0.astype("<f4").tobytes()
    data += w.astype("<f4").tobytes()
    data += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + data)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 per feature). Output record: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C * (1L + (long)d * 256 * (1L + K));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += (long)C * d * 256;
    const float *v = p; /* p += (long)C * d * 256 * K; */

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * C * K);
    float *sum_v_sq = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            logits[c] = w0[c];
            for (int kk = 0; kk < K; kk++) {
                sum_v[c * K + kk] = 0.0f;
                sum_v_sq[c * K + kk] = 0.0f;
            }
        }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++) {
                logits[c] += w[((long)c * d + j) * 256 + val];
                const float *v_ptr = v + (((long)c * d + j) * 256 + val) * K;
                for (int kk = 0; kk < K; kk++) {
                    float vv = v_ptr[kk];
                    sum_v[c * K + kk] += vv;
                    sum_v_sq[c * K + kk] += vv * vv;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sv = sum_v[c * K + kk];
                inter += 0.5f * (sv * sv - sum_v_sq[c * K + kk]);
            }
            logits[c] += inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```