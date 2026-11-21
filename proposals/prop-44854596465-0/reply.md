```hypothesis
Factorization Machine (FM) with class-specific embeddings captures pairwise feature interactions
efficiently. On this tabular task with 14 mixed-type features, low-order interactions are known to
help; FM adds only O(d * 256 * k) parameters (k=4..16) and O(d^2 * k) inference ops per item.
Expect to beat label-linear (0.8524) with similar latency. If dev accuracy ≤ 0.8524, the family
does not justify its complexity.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Chooses hyperparameters on last 20% of train.in,
then trains on all data. Exports weights as: magic, d, k, C; float32 linear[C][d][256],
emb[C][d][256][k]."""
import json, math, struct, sys
import numpy as np

SEED = 1

# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03, 0.1]
L2_CHOICES = [0.001, 0.01]
EPOCHS_CHOICES = [10, 20, 30]
BATCH_CHOICES = [64, 256]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, linear, emb, C, k):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    logits = np.zeros((n, C), np.float32)
    # Linear term
    for c in range(C):
        for j in range(d):
            logits[:, c] += linear[c, j, X[:, j]]
    # Interaction term: 0.5 * ((sum emb)^2 - sum emb^2)
    for c in range(C):
        sum_emb = np.zeros((n, k), np.float32)
        sum_emb_sq = np.zeros((n, k), np.float32)
        for j in range(d):
            e = emb[c, j, X[:, j]]  # (n, k)
            sum_emb += e
            sum_emb_sq += e * e
        inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
        logits[:, c] += inter
    return logits


def fm_backward(X, y, linear, emb, C, k, L2):
    """Compute gradients for linear and emb. Returns (grad_linear, grad_emb)."""
    n, d = X.shape
    logits = fm_forward(X, linear, emb, C, k)
    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # (n, C)

    grad_linear = np.zeros_like(linear)
    grad_emb = np.zeros_like(emb)

    # Linear gradients
    for c in range(C):
        for j in range(d):
            # For each feature value v, sum probs[i, c] where X[i, j] == v
            for v in range(256):
                mask = (X[:, j] == v)
                if np.any(mask):
                    grad_linear[c, j, v] = probs[mask, c].sum()

    # Embedding gradients
    for c in range(C):
        # Compute sum_emb for each example
        sum_emb = np.zeros((n, k), np.float32)
        for j in range(d):
            sum_emb += emb[c, j, X[:, j]]
        # For each feature j, gradient contribution is probs * (sum_emb - emb_j)
        for j in range(d):
            e = emb[c, j, X[:, j]]  # (n, k)
            diff = (sum_emb - e) * probs[:, c][:, None]  # (n, k)
            # Accumulate per feature value
            for v in range(256):
                mask = (X[:, j] == v)
                if np.any(mask):
                    grad_emb[c, j, v] = diff[mask].sum(axis=0)

    # L2 regularization
    grad_linear += L2 * linear
    grad_emb += L2 * emb
    return grad_linear, grad_emb


def train_fm(X, y, C, k, lr, L2, epochs, batch, rng):
    n, d = X.shape
    linear = np.zeros((C, d, 256), np.float32)
    emb = rng.uniform(-0.01, 0.01, (C, d, 256, k)).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            gl, ge = fm_backward(xb, yb, linear, emb, C, k, L2)
            t += 1
            # Update linear
            m_lin = beta1 * m_lin + (1 - beta1) * gl
            v_lin = beta2 * v_lin + (1 - beta2) * gl * gl
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # Update emb
            m_emb = beta1 * m_emb + (1 - beta1) * ge
            v_emb = beta2 * v_emb + (1 - beta2) * ge * ge
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return linear, emb


def evaluate(X, y, linear, emb, C, k):
    logits = fm_forward(X, linear, emb, C, k)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Split train / tune (last 20% for tuning)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_linear = None
    best_emb = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                for epochs in EPOCHS_CHOICES:
                    for batch in BATCH_CHOICES:
                        lin, emb = train_fm(X_tr, y_tr, C, k, lr, L2, epochs, batch, rng)
                        acc = evaluate(X_tu, y_tu, lin, emb, C, k)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = (k, lr, L2, epochs, batch)
                            best_linear, best_emb = lin.copy(), emb.copy()

    # Retrain on full data with best params
    k, lr, L2, epochs, batch = best_params
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_linear, final_emb = train_fm(X, y, C, k, lr, L2, epochs, batch, rng)

    # Write weights.bin
    # Format: magic(0x464D3132), d, k, C (int32 each)
    # Then linear[C][d][256] (float32), then emb[C][d][256][k] (float32)
    magic = 0x464D3132  # 'FM12'
    out = struct.pack("<4i", magic, d, k, C)
    out += final_linear.astype("<f4").tobytes()
    out += final_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C * d * 256 + (long)C * d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    float *sum_emb = malloc(C * k * sizeof(float));
    float *sum_emb_sq = malloc(C * k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = 0.0f;
            for (int l = 0; l < k; l++) {
                sum_emb[c * k + l] = 0.0f;
                sum_emb_sq[c * k + l] = 0.0f;
            }
        }

        // Linear term and accumulate embeddings
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                logits[c] += linear[((long)c * d + j) * 256 + v];
                const float *e = emb + ((long)c * d * 256 + j * 256 + v) * k;
                for (int l = 0; l < k; l++) {
                    float val = e[l];
                    sum_emb[c * k + l] += val;
                    sum_emb_sq[c * k + l] += val * val;
                }
            }
        }

        // Interaction term
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int l = 0; l < k; l++) {
                float s = sum_emb[c * k + l];
                inter += 0.5f * (s * s - sum_emb_sq[c * k + l]);
            }
            logits[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(logits); free(sum_emb); free(sum_emb_sq);
    return 0;
}
```