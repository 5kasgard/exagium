```hypothesis
Family: label-factorized (low-rank factorized linear model).
Why: The linear model has 4096×65 parameters, which may overfit on 24k samples. A rank-D factorization (U: 4096×D, V: D×65) reduces parameters, acts as regularization, and shares statistical strength across positions and classes. For character prediction, nearby contexts share patterns; factorization exploits this. Inference cost is O(16·D + D·C) vs O(4096·C) for linear, so it is faster and smaller for modest D.
Failure indication: If dev accuracy does not exceed label-linear (0.3375) or if no D on the tuning split beats linear, the factorization does not help.
```
```python
"""Factorized linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Factorizes the weight matrix as U (d*256 x D) @ V (D x C) + bias.
Chooses D on the last 20% of train.in (tuning split), then trains on all data.
Exports: magic 0x46435431, d, D, C; float32 U[(d*256)*D], V[D*C], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 30
EPOCHS_FULL = 60
D_CANDIDATES = [8, 16, 32, 64]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]
    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_va = X_bytes[:split], X_bytes[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)

    def train_eval(D, X_tr, y_tr, X_va, y_va, epochs):
        """Train factorized model for given D, return validation accuracy."""
        # Flattened index: position * 256 + byte value
        # U shape (d*256, D), V shape (D, C), bias (C,)
        scale = math.sqrt(6.0 / (d * 256 + D))
        U = (rng.uniform(-1, 1, (d * 256, D)) * scale).astype(np.float32)
        V = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / (D + C))).astype(np.float32)
        bias = np.zeros(C, np.float32)
        params = [U, V, bias]
        # Adam state
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        n_tr = X_tr.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y[idx]
                bsz = xb.shape[0]
                # Compute indices into U: (bsz, d)
                indices = (np.arange(d, dtype=np.int64) * 256 + xb.astype(np.int64))  # (bsz, d)
                # Sum embeddings: (bsz, D)
                u_sum = np.zeros((bsz, D), np.float32)
                for pos in range(d):
                    u_sum += U[indices[:, pos]]
                # Logits: (bsz, C)
                logits = u_sum @ V + bias
                # Softmax + cross-entropy gradient
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits, dtype=np.float32)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(bsz), yb] -= 1.0
                probs /= bsz
                # Gradients
                grad_V = u_sum.T @ probs  # (D, C)
                grad_bias = probs.sum(axis=0)
                grad_u_sum = probs @ V.T  # (bsz, D)
                grad_U = np.zeros_like(U)
                for pos in range(d):
                    np.add.at(grad_U, indices[:, pos], grad_u_sum)
                # L2 regularization
                grad_U += L2 * U
                grad_V += L2 * V
                # Adam update
                t += 1
                grads = [grad_U, grad_V, grad_bias]
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        indices_va = (np.arange(d, dtype=np.int64) * 256 + X_va.astype(np.int64))
        u_sum_va = np.zeros((X_va.shape[0], D), np.float32)
        for pos in range(d):
            u_sum_va += U[indices_va[:, pos]]
        logits_va = u_sum_va @ V + bias
        preds = logits_va.argmax(axis=1)
        acc = (preds == y_va).mean()
        return acc, (U, V, bias)

    # Tune D
    best_D, best_acc, best_params = None, -1, None
    for D in D_CANDIDATES:
        acc, params = train_eval(D, X_tr, y_tr, X_va, y_va, EPOCHS_TUNE)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params
    # Retrain on full data with best D
    D = best_D
    U, V, bias = best_params
    # Re-initialize with same seed for reproducibility? We'll continue from best_params but reset Adam state.
    # Instead, retrain from scratch on full data for more epochs.
    scale = math.sqrt(6.0 / (d * 256 + D))
    U = (rng.uniform(-1, 1, (d * 256, D)) * scale).astype(np.float32)
    V = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / (D + C))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [U, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(EPOCHS_FULL):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_bytes[idx], y[idx]
            bsz = xb.shape[0]
            indices = (np.arange(d, dtype=np.int64) * 256 + xb.astype(np.int64))
            u_sum = np.zeros((bsz, D), np.float32)
            for pos in range(d):
                u_sum += U[indices[:, pos]]
            logits = u_sum @ V + bias
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz
            grad_V = u_sum.T @ probs
            grad_bias = probs.sum(axis=0)
            grad_u_sum = probs @ V.T
            grad_U = np.zeros_like(U)
            for pos in range(d):
                np.add.at(grad_U, indices[:, pos], grad_u_sum)
            grad_U += L2 * U
            grad_V += L2 * V
            t += 1
            grads = [grad_U, grad_V, grad_bias]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x46435431  # 'FCT1'
    out = struct.pack("<4i", magic, d, D, C)
    out += U.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorized linear model inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic(0x46435431), d, D, C; float32 U[d*256*D], V[D*C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x46435431) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *U = (const float *)(hd + 4);
    const float *V = U + (long)d * 256 * D;
    const float *bias = V + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *u_sum = (float *)malloc(sizeof(float) * D);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // zero u_sum
        for (int j = 0; j < D; j++) u_sum[j] = 0.0f;
        // accumulate embeddings
        for (int p = 0; p < d; p++) {
            const float *u_row = U + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) u_sum[j] += u_row[j];
        }
        // compute scores = u_sum * V + bias
        for (int c = 0; c < C; c++) {
            float sum = bias[c];
            for (int j = 0; j < D; j++) sum += u_sum[j] * V[(long)j * C + c];
            scores[c] = sum;
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(u_sum);
    free(scores);
    return 0;
}
```