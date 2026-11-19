```hypothesis
Family: label-fm (factorization machine with shared embeddings).
Why: Adult has 14 features (mix of ordinal/categorical) where pairwise interactions matter (e.g., education×occupation). A rank-4 FM adds only ~14k parameters and ~400 FLOPs/item, capturing interactions missed by label-linear while staying far cheaper than label-mlp16 or label-gbdt. Hyperparameters (E, lr, L2, epochs) are chosen on the last 20% tuning split.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000 (≈1.4× label-linear), the model does not meet the goal.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Encodes each feature value (0..255) with a learned embedding (size E) and a linear weight.
Shared embeddings across classes, class-specific linear weights and biases.
Training: Adam on log-loss, minibatch 64, early stopping on tuning split (last 20% of train.in).
Weights format (little-endian):
  int32 magic=0x464D4C31 ('FML1'), int32 d, int32 C, int32 E
  float32 bias[C]
  float32 linear[C][d][256]
  float32 embed[d][256][E]
"""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS = 200
PATIENCE = 10
TUNE_FRAC = 0.2

def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X_bytes = b"".join(read_records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning (by file order)
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    n_tr = X_tr.shape[0]

    # Hyperparameter search on tuning split
    best_acc = -1.0
    best_cfg = None
    best_params = None
    rng = np.random.default_rng(SEED)

    for E in (2, 4, 8):
        for lr in (1e-2, 3e-3, 1e-3):
            for L2 in (1e-4, 1e-5, 0.0):
                # Initialize parameters
                bias = np.zeros(C, np.float32)
                linear = np.zeros((C, d, 256), np.float32)
                embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)

                # Adam state
                m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
                m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
                m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
                t = 0

                def predict_logits(Xb):
                    # Xb: (B, d) uint8
                    B = Xb.shape[0]
                    logits = np.zeros((B, C), np.float32)
                    # Linear part
                    for c in range(C):
                        logits[:, c] = bias[c]
                        for j in range(d):
                            logits[:, c] += linear[c, j, Xb[:, j]]
                    # Interaction part: sum_{i<j} <emb[i], emb[j]>
                    if E > 0:
                        # Compute summed embeddings per example
                        sum_emb = np.zeros((B, E), np.float32)
                        sum_sq_emb = np.zeros((B, E), np.float32)
                        for j in range(d):
                            e = embed[j, Xb[:, j], :]  # (B, E)
                            sum_emb += e
                            sum_sq_emb += e * e
                        inter = 0.5 * (sum_emb * sum_emb - sum_sq_emb).sum(axis=1)  # (B,)
                        for c in range(C):
                            logits[:, c] += inter
                    return logits

                def accuracy(Xb, yb):
                    logits = predict_logits(Xb)
                    pred = logits.argmax(axis=1)
                    return (pred == yb).mean()

                best_te_acc = -1.0
                no_improve = 0
                best_state = None

                for epoch in range(MAX_EPOCHS):
                    order = rng.permutation(n_tr)
                    for s in range(0, n_tr, BATCH):
                        idx = order[s:s + BATCH]
                        Xb, yb = X_tr[idx], y_tr[idx]
                        B = Xb.shape[0]

                        # Forward
                        logits = np.zeros((B, C), np.float32)
                        sum_emb = np.zeros((B, E), np.float32)
                        sum_sq_emb = np.zeros((B, E), np.float32)
                        for c in range(C):
                            logits[:, c] = bias[c]
                            for j in range(d):
                                logits[:, c] += linear[c, j, Xb[:, j]]
                        for j in range(d):
                            e = embed[j, Xb[:, j], :]
                            sum_emb += e
                            sum_sq_emb += e * e
                        inter = 0.5 * (sum_emb * sum_emb - sum_sq_emb).sum(axis=1)
                        for c in range(C):
                            logits[:, c] += inter

                        # Softmax + cross-entropy gradient
                        logits -= logits.max(axis=1, keepdims=True)
                        exp_logits = np.exp(logits)
                        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                        grad_logits = probs
                        grad_logits[np.arange(B), yb] -= 1.0
                        grad_logits /= B

                        # Gradients
                        g_bias = grad_logits.sum(axis=0)  # (C,)
                        g_lin = np.zeros_like(linear)
                        g_emb = np.zeros_like(embed)

                        for c in range(C):
                            gc = grad_logits[:, c]
                            for j in range(d):
                                # linear gradient
                                np.add.at(g_lin[c, j], Xb[:, j], gc)
                            # interaction gradient: d(inter)/d(emb[j,v]) = sum_emb - emb[j,v]
                            if E > 0:
                                for j in range(d):
                                    # sum_emb is (B,E), emb[j, Xb[:,j]] is (B,E)
                                    diff = sum_emb - embed[j, Xb[:, j], :]
                                    # multiply by gc[:,None] and accumulate
                                    np.add.at(g_emb[j], Xb[:, j], (gc[:, None] * diff).astype(np.float32))

                        # L2 regularization
                        g_lin += L2 * linear
                        g_emb += L2 * embed

                        # Adam update
                        t += 1
                        beta1, beta2, eps = 0.9, 0.999, 1e-8
                        # bias
                        m_bias = beta1 * m_bias + (1 - beta1) * g_bias
                        v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
                        m_hat = m_bias / (1 - beta1 ** t)
                        v_hat = v_bias / (1 - beta2 ** t)
                        bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
                        # linear
                        m_lin = beta1 * m_lin + (1 - beta1) * g_lin
                        v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
                        m_hat = m_lin / (1 - beta1 ** t)
                        v_hat = v_lin / (1 - beta2 ** t)
                        linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
                        # embed
                        m_emb = beta1 * m_emb + (1 - beta1) * g_emb
                        v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
                        m_hat = m_emb / (1 - beta1 ** t)
                        v_hat = v_emb / (1 - beta2 ** t)
                        embed -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    # Evaluate on tuning split
                    te_acc = accuracy(X_te, y_te)
                    if te_acc > best_te_acc:
                        best_te_acc = te_acc
                        no_improve = 0
                        best_state = (bias.copy(), linear.copy(), embed.copy())
                    else:
                        no_improve += 1
                        if no_improve >= PATIENCE:
                            break

                if best_te_acc > best_acc:
                    best_acc = best_te_acc
                    best_cfg = (E, lr, L2)
                    best_params = best_state

    # Retrain on full training set with best config
    E, lr, L2 = best_cfg
    bias, linear, embed = best_params
    # Continue training on full data for a few more epochs (optional, but we can just use the best_state from tuning)
    # To be safe, we do a few epochs on full data starting from best_state.
    n_full = X.shape[0]
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0
    rng_full = np.random.default_rng(SEED + 123)

    for epoch in range(20):  # fixed small number of epochs on full data
        order = rng_full.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            B = Xb.shape[0]

            # Forward
            logits = np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, E), np.float32)
            sum_sq_emb = np.zeros((B, E), np.float32)
            for c in range(C):
                logits[:, c] = bias[c]
                for j in range(d):
                    logits[:, c] += linear[c, j, Xb[:, j]]
            for j in range(d):
                e = embed[j, Xb[:, j], :]
                sum_emb += e
                sum_sq_emb += e * e
            inter = 0.5 * (sum_emb * sum_emb - sum_sq_emb).sum(axis=1)
            for c in range(C):
                logits[:, c] += inter

            # Gradients
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            g_bias = grad_logits.sum(axis=0)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)

            for c in range(C):
                gc = grad_logits[:, c]
                for j in range(d):
                    np.add.at(g_lin[c, j], Xb[:, j], gc)
                if E > 0:
                    for j in range(d):
                        diff = sum_emb - embed[j, Xb[:, j], :]
                        np.add.at(g_emb[j], Xb[:, j], (gc[:, None] * diff).astype(np.float32))

            g_lin += L2 * linear
            g_emb += L2 * embed

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= lr * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, C, E)
    out = bytearray()
    out += header
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0..C-1).
 * Weights format (little-endian):
 *   int32 magic=0x464D4C31, int32 d, int32 C, int32 E
 *   float32 bias[C]
 *   float32 linear[C][d][256]
 *   float32 embed[d][256][E]
 * Inference: logits_c = bias_c + sum_j linear[c][j][x_j] + sum_{i<j} <embed[i][x_i], embed[j][x_j]>
 *            = bias_c + linear_sum_c + 0.5*( (sum_emb)^2 - sum(emb^2) ) summed over E
 * Output: argmax_c logits_c (first max wins). */
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
    int d = hd[1], C = hd[2], E = hd[3];
    if (C < 1 || C > 256 || E < 0 || E > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * 256 + 4L * d * 256 * E;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * E);
    float *sum_sq_emb = (float *)malloc(sizeof(float) * E);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize logits with bias
        for (int c = 0; c < C; c++) logits[c] = bias[c];

        // Linear part
        for (int c = 0; c < C; c++) {
            const float *lin_c = linear + (long)c * d * 256;
            for (int j = 0; j < d; j++) {
                logits[c] += lin_c[(long)j * 256 + xi[j]];
            }
        }

        // Interaction part (factorization machine)
        if (E > 0) {
            for (int e = 0; e < E; e++) {
                sum_emb[e] = 0.0f;
                sum_sq_emb[e] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                const float *emb_j = embed + ((long)j * 256 + xi[j]) * E;
                for (int e = 0; e < E; e++) {
                    float v = emb_j[e];
                    sum_emb[e] += v;
                    sum_sq_emb[e] += v * v;
                }
            }
            float inter = 0.0f;
            for (int e = 0; e < E; e++) {
                float s = sum_emb[e];
                inter += 0.5f * (s * s - sum_sq_emb[e]);
            }
            for (int c = 0; c < C; c++) logits[c] += inter;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(sum_sq_emb); free(logits);
    return 0;
}
```