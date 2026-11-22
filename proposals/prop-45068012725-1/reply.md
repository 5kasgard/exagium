```hypothesis
Family: label-fm (factorization machine, degree 2, low-rank k=4).
Why: Optdigits pixels have strong pairwise interactions (strokes, loops) that a linear model misses.
A factorization machine captures all pairwise interactions with O(d*k) parameters (d=64, k=4 → ~2.5k params),
staying tiny and fast. It should beat label-linear (96.1%) and approach label-mlp16 (97.7%) at lower inference cost.
Failure: If dev accuracy ≤ 96.5% or instructions/item > 10k, the family is not worthwhile.
```
```python
"""label-fm: Factorization Machine (degree 2, rank k) for fixed-size ordinal bytes.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Weights format (little-endian):
  int32 magic=0x464D4C31, d, k, C
  float32 mu[d], sd[d]
  float32 w[C][d]
  float32 V[C][d][k]
  float32 bias[C]
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
K = 4
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 128
MAX_EPOCHS = 50
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Split: last 20% for tuning
    split = int(n * 0.8)
    idx = np.arange(n)
    tr_idx, val_idx = idx[:split], idx[split:]

    # Normalization on training portion
    mu = X_bytes[tr_idx].mean(axis=0).astype(np.float32)
    sd = X_bytes[tr_idx].std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    def normalize(X):
        return ((X.astype(np.float32) - mu) / sd).astype(np.float32)

    X_tr = normalize(X_bytes[tr_idx])
    X_val = normalize(X_bytes[val_idx])
    y_tr, y_val = y[tr_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    w = (rng.normal(0, 0.01, (C, d)) * math.sqrt(2.0 / d)).astype(np.float32)
    V = (rng.normal(0, 0.01, (C, d, K)) * math.sqrt(2.0 / d)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    params = [w, V, bias]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_params = None
    wait = 0

    def predict_scores(X):
        # X: (B, d) normalized
        # linear: (B, C)
        lin = X @ w.T + bias  # (B, C)
        # interaction: 0.5 * [ (sum_i V[i] x_i)^2 - sum_i (V[i]^2) x_i^2 ]
        # Compute for each class separately for memory efficiency
        inter = np.zeros((X.shape[0], C), np.float32)
        for c in range(C):
            Vc = V[c]  # (d, K)
            # X @ Vc -> (B, K)
            xv = X @ Vc
            # sum_i (Vc[i]^2) * X[:,i]^2
            Vc_sq = (Vc * Vc).sum(axis=1)  # (d,)
            x_sq = X * X
            inter[:, c] = 0.5 * ((xv * xv).sum(axis=1) - x_sq @ Vc_sq)
        return lin + inter

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), BATCH):
            batch_idx = order[s:s+BATCH]
            Xb, yb = X_tr[batch_idx], y_tr[batch_idx]
            scores = predict_scores(Xb)
            grad = softmax_cross_entropy_grad(scores, yb)  # (B, C)

            # Gradients
            # linear: Xb.T @ grad  -> (d, C) -> transpose to (C, d)
            gw = Xb.T @ grad + L2 * w
            gw = gw.T.astype(np.float32)

            # V: for each class c, Vc grad = Xb.T @ (grad[:,c][:,None] * (xv_c)) - Xb.T @ (grad[:,c][:,None] * (Xb * Vc_sq?))
            # Let's derive: inter_c = 0.5 * ( (X Vc)^2 - sum_i Vc[i]^2 X_i^2 )
            # d inter_c / d Vc = (X Vc) X^T - X^T (X^2 * Vc_sq)  (where * is elementwise)
            # But grad is dL/d score, so dL/d Vc = grad_c[:,None] * d inter_c / d Vc
            gV = np.zeros_like(V)
            for c in range(C):
                Vc = V[c]
                xv = Xb @ Vc  # (B, K)
                Vc_sq = (Vc * Vc).sum(axis=1)  # (d,)
                # term1: (grad_c * xv).T @ Xb  -> (K, B) @ (B, d) = (K, d) -> transpose to (d, K)
                term1 = (grad[:, c:c+1] * xv).T @ Xb  # (K, d)
                # term2: (grad_c * (Xb^2 @ Vc_sq))[:,None] * Xb? Wait.
                # d/d Vc of sum_i Vc[i]^2 X_i^2 = 2 * Vc * X_i^2 (for each i)
                # So term2 = Xb.T @ (grad_c[:,None] * (Xb * (2*Vc)?) ) ... let's do elementwise:
                # For each feature i: contribution to Vc[i] is -0.5 * grad_c * X_i^2 * 2 * Vc[i] = - grad_c * X_i^2 * Vc[i]
                # So term2 = (Xb * Xb).T @ (grad[:,c:c+1] * Vc)  # (d, B) @ (B, K) = (d, K)
                term2 = (Xb * Xb).T @ (grad[:, c:c+1] * Vc)
                gV[c] = (term1.T - term2) + L2 * Vc

            gbias = grad.sum(axis=0).astype(np.float32)

            grads = [gw, gV, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation
        val_scores = predict_scores(X_val)
        val_pred = val_scores.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Retrain on full training set with best hyperparameters (same epochs as best)
    # For simplicity, we just use the best_params found (they were trained on 80%).
    # The rules say: choose settings on tuning split, then train on all of train.
    # We'll do a quick retrain on full data starting from best_params.
    X_full = normalize(X_bytes)
    w, V, bias = best_params
    params = [w, V, bias]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s+BATCH]
            Xb, yb = X_full[batch_idx], y[batch_idx]
            scores = predict_scores(Xb)
            grad = softmax_cross_entropy_grad(scores, yb)
            gw = Xb.T @ grad + L2 * w
            gw = gw.T.astype(np.float32)
            gV = np.zeros_like(V)
            for c in range(C):
                Vc = V[c]
                xv = Xb @ Vc
                Vc_sq = (Vc * Vc).sum(axis=1)
                term1 = (grad[:, c:c+1] * xv).T @ Xb
                term2 = (Xb * Xb).T @ (grad[:, c:c+1] * Vc)
                gV[c] = (term1.T - term2) + L2 * Vc
            gbias = grad.sum(axis=0).astype(np.float32)
            grads = [gw, gV, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization machine (degree 2, rank k). Input: d bytes (0..255). Output: one byte class.
 * Weights layout (little-endian):
 *   int32 magic=0x464D4C31, d, k, C
 *   float32 mu[d], sd[d]
 *   float32 w[C][d]
 *   float32 V[C][d][k]
 *   float32 bias[C] */
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
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 2 * sizeof(float)           // mu, sd
        + (size_t)C * d * sizeof(float)           // w
        + (size_t)C * d * k * sizeof(float)       // V
        + (size_t)C * sizeof(float);              // bias
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *w = sd + d;
    const float *V = w + (size_t)C * d;
    const float *bias = V + (size_t)C * d * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float best_score = -1e30f; unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            // Linear part
            float score = bias[c];
            const float *wc = w + (size_t)c * d;
            const float *Vc = V + (size_t)c * d * k;
            // Accumulate xv = sum_i Vc[i] * x_norm_i
            float xv[16]; // k <= 16 (we use k=4)
            for (int j = 0; j < k; j++) xv[j] = 0.0f;

            for (int j = 0; j < d; j++) {
                float xn = (xi[j] - mu[j]) / sd[j];
                score += wc[j] * xn;
                for (int jj = 0; jj < k; jj++) {
                    xv[jj] += Vc[(size_t)j * k + jj] * xn;
                }
            }
            // Interaction part: 0.5 * (sum xv^2 - sum_i (Vc[i]^2) * xn^2)
            float inter = 0.0f;
            for (int jj = 0; jj < k; jj++) inter += xv[jj] * xv[jj];
            for (int j = 0; j < d; j++) {
                float Vc_sq = 0.0f;
                for (int jj = 0; jj < k; jj++) {
                    float v = Vc[(size_t)j * k + jj];
                    Vc_sq += v * v;
                }
                float xn = (xi[j] - mu[j]) / sd[j];
                inter -= Vc_sq * xn * xn;
            }
            score += 0.5f * inter;

            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```