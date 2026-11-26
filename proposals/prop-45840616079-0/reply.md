```hypothesis
Factorization Machine (FM) with latent dim k=4 captures pairwise feature interactions (e.g., education×occupation, capital-gain×hours) vital for tabular data, while inference stays O(d·k) ≈ 56 FMA per class. Beats label-linear (0.8524) by modeling interactions linearly; if it fails, interactions need higher rank or are negligible here.
```
```python
"""Factorization Machine for tabular label tasks: per-field embeddings + linear term.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Input: 14 bytes (0-255 each) per record. Model: w0[c] + sum_i w[c,i,xi] + 0.5*sum_f((sum_i V[i,xi,f])^2 - sum_i V[i,xi,f]^2).
Exported: int32 magic 0x464D2031, d, k, C; float32 w0[C], w[C,d,256], V[d,256,k]."""
import json, math, struct, sys

import numpy as np

# Tuned on last 20% of train.in (held-out order): k=4, lr=1e-2, l2=1e-4, batch=256, epochs~23 (600k samples)
SEED, LR, L2, BATCH, TARGET_SAMPLES, K = 1, 1e-2, 1e-4, 256, 600000, 4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X_bytes.shape
    C = card["classes"]
    assert d == 14 and C == 2

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_va, y_va = X_bytes[split:], y[split:]
    n_tr = len(X_tr)

    rng = np.random.default_rng(SEED)

    # Params: w0[C], w[C,d,256], V[d,256,k]
    w0 = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (C, d, 256)) * 0).astype(np.float32)  # zero init for linear
    V = (rng.normal(0, 0.01, (d, 256, K)) * 1).astype(np.float32)  # small random for embeddings

    # Adam state
    params = [w0, w, V]
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0

    def predict_scores(Xb):
        """Xb: (m, d) uint8 -> scores (m, C)"""
        m = Xb.shape[0]
        scores = np.tile(w0, (m, 1))  # (m, C)
        # Linear term
        for i in range(d):
            scores += w[:, i, Xb[:, i]].T  # (m, C)
        # FM interaction term (shared V across classes)
        for f in range(K):
            v_sum = np.zeros(m, np.float32)
            v_sq_sum = np.zeros(m, np.float32)
            for i in range(d):
                v = V[i, Xb[:, i], f]
                v_sum += v
                v_sq_sum += v * v
            inter = 0.5 * (v_sum * v_sum - v_sq_sum)  # (m,)
            scores += inter[:, None]  # broadcast to C
        return scores

    def loss_grad(Xb, yb):
        m = len(yb)
        scores = predict_scores(Xb)
        # Softmax + cross-entropy gradient
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs[np.arange(m), yb] -= 1.0
        probs /= m  # (m, C)

        # Gradients
        gw0 = probs.sum(axis=0).astype(np.float32)  # (C,)
        gw = np.zeros((C, d, 256), np.float32)
        gV = np.zeros((d, 256, K), np.float32)

        # Linear grad: for each feature i, value v, add probs[:,c] to gw[c,i,v]
        for i in range(d):
            vals = Xb[:, i]
            for c in range(C):
                np.add.at(gw[c, i], vals, probs[:, c])

        # FM grad: d/dV[i,v,f] = sum_f (sum_j V[j,xj,f] - V[i,v,f]) * prob_c
        # For each factor f, compute sum_j V[j,xj,f] per sample
        for f in range(K):
            v_sum = np.zeros(m, np.float32)
            for i in range(d):
                v_sum += V[i, Xb[:, i], f]
            for i in range(d):
                vals = Xb[:, i]
                vi = V[i, vals, f]  # (m,)
                # grad = (v_sum - vi) * probs[:,c] summed over c? Wait, interaction term added to ALL classes equally.
                # Since interaction is shared, gradient w.r.t V is sum_c probs[:,c] * (v_sum - vi)
                grad_factor = (v_sum - vi) * probs.sum(axis=1)  # (m,)
                np.add.at(gV[i, :, f], vals, grad_factor)

        return [gw0, gw, gV], probs

    epochs = min(100, math.ceil(TARGET_SAMPLES / n_tr))
    best_acc = -1
    best_params = None

    for ep in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            grads, _ = loss_grad(Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # L2 regularization (weight decay)
                if i > 0:  # skip w0
                    p -= LR * L2 * p

        # Validation accuracy
        va_scores = predict_scores(X_va)
        va_pred = va_scores.argmax(axis=1)
        acc = (va_pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]

    # Retrain on full data with same epochs (deterministic)
    # Reset params and optimizer
    w0, w, V = [rng.normal(0, 0.01, p.shape).astype(np.float32) * (0 if i==0 else 1) for i, p in enumerate(params)]
    w0[:] = 0
    w[:] = 0
    state_m = [np.zeros_like(p) for p in [w0, w, V]]
    state_v = [np.zeros_like(p) for p in [w0, w, V]]
    t = 0

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            grads, _ = loss_grad(Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip([w0, w, V], grads)):
                m, v = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                if i > 0:
                    p -= LR * L2 * p

    # Pack weights: magic, d, k, C; w0[C], w[C,d,256], V[d,256,k]
    out = struct.pack("<4i", 0x464D2031, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (0-255 each). Output: 1 byte class (0 or 1).
 * Weights: magic 0x464D2031, d, k, C; float32 w0[C], w[C,d,256], V[d,256,k]. */
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
    if (nw < 16 || hd[0] != 0x464D2031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k < 1 || k > 16) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w = w0 + C;
    const float *V = w + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear term
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) scores[c] += w[((long)c * d + j) * 256 + v];
        }

        // FM interaction (shared V)
        for (int f = 0; f < k; f++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * k + f];
                sum += v;
                sum_sq += v * v;
            }
            float inter = 0.5f * (sum * sum - sum_sq);
            for (int c = 0; c < C; c++) scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```