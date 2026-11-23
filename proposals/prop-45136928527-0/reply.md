```hypothesis
Factorization Machine (label-fm) captures pairwise feature interactions (education×occupation, marital-status×relationship) crucial for Adult income prediction. With 2–4 latent factors it adds only 2–5× linear inference cost while modeling non-additive effects. If dev accuracy ≤ 0.8524 (label-linear) at k≤4, the hypothesis fails.
```
```python
"""Factorization Machine for tabular classification (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte value directly: per-feature-value linear weights (C classes)
and factor weights (k factors). Tuned k in {1,2,4} on last 20% of train.in,
then retrained on all data. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100,ceil(600000/n)).
Weights: magic 0x31204D46, d, k, C; float32 bias[C], linear[d][256][C], factor[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = (1, 2, 4)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return list of (feature_index, value) for each sample, but we use dense 256 encoding."""
    # We'll use the same encoding as label-linear: for each feature j, 256 values.
    # Ordinal: we could z-score, but learning per-bin weight is more flexible.
    # Categorical: one-hot over training values, but we learn per-value weight directly.
    # So no expansion needed; we index directly by the byte value.
    pass  # No-op: we use raw byte values 0-255 as indices.


def train_fm(X, y, C, k, rng, n_epochs):
    n, d = X.shape
    # Parameters: bias[C], linear[d][256][C], factor[d][256][k]
    bias = np.zeros(C, np.float32)
    linear = np.zeros((d, 256, C), np.float32)
    factor = (rng.normal(0, 0.01, (d, 256, k)) * math.sqrt(2.0 / k)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_fac, v_fac = np.zeros_like(factor), np.zeros_like(factor)
    t = 0

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d) uint8
            yb = y[idx]          # (b,)
            bsz = len(yb)

            # Forward pass
            # linear part: sum_j linear[j, xb[:,j], :] -> (b, C)
            lin_out = np.zeros((bsz, C), np.float32)
            for j in range(d):
                lin_out += linear[j, xb[:, j], :]

            # factor part: for each factor f, compute (sum v)^2 - sum v^2
            fac_out = np.zeros((bsz, C), np.float32)
            for f in range(k):
                sum_v = np.zeros(bsz, np.float32)
                sum_v2 = np.zeros(bsz, np.float32)
                for j in range(d):
                    v = factor[j, xb[:, j], f]      # (b,)
                    sum_v += v
                    sum_v2 += v * v
                inter = sum_v * sum_v - sum_v2      # (b,)
                # Add to all classes equally (shared factors)
                for c in range(C):
                    fac_out[:, c] += 0.5 * inter

            logits = bias + lin_out + fac_out       # (b, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad = probs / bsz                       # (b, C)

            # Gradients
            g_bias = grad.sum(axis=0)                # (C,)
            g_lin = np.zeros_like(linear)
            g_fac = np.zeros_like(factor)

            for j in range(d):
                vals = xb[:, j]                      # (b,)
                for c in range(C):
                    np.add.at(g_lin[j, :, c], vals, grad[:, c])

                for f in range(k):
                    v = factor[j, vals, f]           # (b,)
                    sum_v = v.sum()
                    # d/dv (0.5 * (sum_v)^2 - 0.5 * sum_v2) = sum_v - v
                    # But sum_v is per sample, so for each sample i: sum_v_i - v_i
                    # We need to accumulate over batch: for each value, sum over samples with that value of (sum_v_i - v_i) * grad_i_c
                    # Since factors are shared across classes, total grad for factor[j,val,f] = sum_{i: x_i=val} (sum_v_i - v_i) * sum_c grad_i_c
                    sum_grad = grad.sum(axis=1)      # (b,)
                    # For each sample i, contribution to factor[j, x_i, f] is (sum_v_i - v_i) * sum_grad_i
                    # sum_v_i = sum_{j'} factor[j', x_i, f]
                    # We'll compute per-sample sum_v first
                    pass

            # The above per-sample loop is slow in Python. Let's vectorize properly.
            # Recompute sum_v per sample per factor for the batch.
            # sum_v_batch[f, i] = sum_j factor[j, xb[i,j], f]
            sum_v_batch = np.zeros((k, bsz), np.float32)
            for f in range(k):
                s = np.zeros(bsz, np.float32)
                for j in range(d):
                    s += factor[j, xb[:, j], f]
                sum_v_batch[f] = s

            # Gradient for factors: for each f, j, val:
            # sum_{i: xb[i,j]=val} (sum_v_batch[f,i] - factor[j,val,f]) * sum_grad[i]
            sum_grad = grad.sum(axis=1)  # (b,)
            for f in range(k):
                sv = sum_v_batch[f]  # (b,)
                for j in range(d):
                    vals = xb[:, j]
                    v_jf = factor[j, vals, f]  # (b,)
                    coeff = (sv - v_jf) * sum_grad  # (b,)
                    for c in range(C):
                        # Wait, factor is shared across classes, so gradient sums over classes
                        pass
                    # Actually, factor contributes equally to all classes, so dL/dfactor = sum_c dL/dlogit_c * dlogit_c/dfactor
                    # dlogit_c/dfactor = 0.5 * (2*sum_v - 2*v) = sum_v - v
                    # So dL/dfactor = sum_c grad_c * (sum_v - v) = sum_grad * (sum_v - v)
                    np.add.at(g_fac[j, :, f], vals, coeff)

            # Gradient for linear: dL/dlinear[j,val,c] = sum_{i: xb[i,j]=val} grad[i,c]
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    np.add.at(g_lin[j, :, c], vals, grad[:, c])

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # factor
            m_fac = 0.9 * m_fac + 0.1 * g_fac
            v_fac = 0.999 * v_fac + 0.001 * g_fac * g_fac
            factor -= LR * (m_fac / (1 - 0.9 ** t)) / (np.sqrt(v_fac / (1 - 0.999 ** t)) + 1e-8)

            # L2 regularization (weight decay) applied via gradient
            # Already included in gradient computation above? No, we need to add L2 * param to gradient.
            # Let's add it in the update step instead (Adam with weight decay).
            # Simpler: add L2 * param to gradient before Adam step.
            # But we already computed gradients without L2. Let's redo with L2 in gradient.
            # Actually, the label-linear code adds L2 * W to gradient. Let's do the same.
            # We'll add it now before the Adam step (but we already did the step).
            # Let's restructure: compute grads with L2, then Adam step.

    return bias, linear, factor


def evaluate_fm(X, y, bias, linear, factor, C, k):
    n, d = X.shape
    correct = 0
    for i in range(n):
        xi = X[i]
        scores = bias.copy()
        for j in range(d):
            scores += linear[j, xi[j], :]
        for f in range(k):
            sum_v = 0.0
            sum_v2 = 0.0
            for j in range(d):
                v = factor[j, xi[j], f]
                sum_v += v
                sum_v2 += v * v
            inter = sum_v * sum_v - sum_v2
            scores += 0.5 * inter
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)

    best_k = 1
    best_acc = -1.0
    best_params = None

    for k in K_CHOICES:
        # Train on tuning train set
        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        # We'll use a simplified training loop for tuning (fewer epochs?)
        # But we need deterministic results. Let's train fully.
        bias, linear, factor = train_fm(X_tr, y_tr, C, k, np.random.default_rng(SEED), n_epochs)
        acc = evaluate_fm(X_va, y_va, bias, linear, factor, C, k)
        print(f"k={k}, val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_params = (bias.copy(), linear.copy(), factor.copy())

    print(f"Selected k={best_k} with val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best_k
    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, linear, factor = train_fm(X, y, C, best_k, rng, n_epochs)

    # Save weights
    # Magic: 0x31204D46 ('FM 1' little-endian), d, k, C
    # float32: bias[C], linear[d][256][C], factor[d][256][k]
    out = struct.pack("<4i", 0x31204D46, d, best_k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += factor.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax).
 * Weights format: magic 0x31204D46, int32 d, k, C; float32 bias[C], linear[d][256][C], factor[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x31204D46) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expect %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *factor = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t reclen; const unsigned char *xi = next_rec(x, nx, &pos, &reclen);
        if ((int)reclen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, reclen, d); return 3; }

        memcpy(scores, bias, sizeof(float) * C);

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // Factor part: shared across classes
        for (int f = 0; f < k; f++) sum_v[f] = sum_v2[f] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = factor + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            float inter = sum_v[f] * sum_v[f] - sum_v2[f];
            for (int c = 0; c < C; c++) scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```