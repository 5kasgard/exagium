```hypothesis
Family: label-fm (factorization machine with pairwise interactions).  
Why: Adult has 14 mixed-type features; FM captures feature interactions efficiently via low-rank embeddings, known to excel on tabular data like this. It adds only ~35k parameters (k=8) over linear, so training stays fast and inference is O(d*k) per item.  
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported weights: magic 0x464D4C31, d, C, k; float32 bias[C], linear[d][256][C], embed[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.005]
L2_CHOICES = [0.001, 0.0001]
BATCH = 256
TUNE_EPOCHS = 10
FINAL_EPOCHS = 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, bias, linear, embed, C, k):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    logits = np.broadcast_to(bias, (n, C)).copy()  # (n, C)
    # linear part
    for j in range(d):
        logits += linear[j, X[:, j], :]  # (n, C)
    # interaction part per class using FM trick
    for c in range(C):
        # gather embeddings for this class: (n, d, k)
        gathered = np.stack([embed[c, j, X[:, j], :] for j in range(d)], axis=1)
        sum_emb = gathered.sum(axis=1)                 # (n, k)
        square_sum = (sum_emb ** 2).sum(axis=1)        # (n,)
        sum_square = (gathered ** 2).sum(axis=(1, 2))  # (n,)
        inter = 0.5 * (square_sum - sum_square)        # (n,)
        logits[:, c] += inter
    return logits


def train_model(X, y, C, k, lr, l2, epochs, rng):
    n, d = X.shape
    # Initialize parameters
    bias = np.zeros(C, np.float32)
    linear = rng.uniform(-0.01, 0.01, (d, 256, C)).astype(np.float32)
    embed = rng.uniform(-0.01, 0.01, (C, d, 256, k)).astype(np.float32)
    params = [bias, linear, embed]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            # Forward
            logits = fm_forward(Xb, bias, linear, embed, C, k)
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)  # (batch, C)
            # Gradients
            g_bias = grad_logits.sum(axis=0)  # (C,)
            g_linear = np.zeros_like(linear)
            g_embed = np.zeros_like(embed)
            # Linear gradient
            for j in range(d):
                np.add.at(g_linear[j], Xb[:, j], grad_logits)
            # Embedding gradient using FM trick
            for c in range(C):
                gc = grad_logits[:, c]  # (batch,)
                # gathered: (batch, d, k)
                gathered = np.stack([embed[c, j, Xb[:, j], :] for j in range(d)], axis=1)
                sum_emb = gathered.sum(axis=1)  # (batch, k)
                # d(inter)/d(emb_{j}) = gc * (sum_emb - emb_{j})
                for j in range(d):
                    diff = (sum_emb - gathered[:, j, :]) * gc[:, None]  # (batch, k)
                    np.add.at(g_embed[c, j], Xb[:, j], diff)
            # Add L2 regularization
            g_bias += l2 * bias
            g_linear += l2 * linear
            g_embed += l2 * embed
            grads = [g_bias, g_linear, g_embed]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return bias, linear, embed


def evaluate(X, y, bias, linear, embed, C, k):
    logits = fm_forward(X, bias, linear, embed, C, k)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X_all)
    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search on tuning split
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                bias, linear, embed = train_model(X_train, y_train, C, k, lr, l2, TUNE_EPOCHS, rng)
                acc = evaluate(X_tune, y_tune, bias, linear, embed, C, k)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (k, lr, l2)
                    best_params = (bias.copy(), linear.copy(), embed.copy())

    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    bias, linear, embed = train_model(X_all, y_all, C, k, lr, l2, FINAL_EPOCHS, rng)

    # Pack weights
    # Order: magic, d, C, k, bias[C], linear[d][256][C], embed[C][d][256][k]
    magic = 0x464D4C31
    header = struct.pack("<4i", magic, d, C, k)
    bias_bytes = bias.astype("<f4").tobytes()
    linear_bytes = linear.astype("<f4").tobytes()
    embed_bytes = embed.astype("<f4").tobytes()
    out = header + bias_bytes + linear_bytes + embed_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index.
 * Weights format: magic 0x464D4C31, int32 d, C, k; float32 bias[C], linear[d][256][C], embed[C][d][256][k]. */
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
    int d = hd[1], C = hd[2], k = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)C * d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    float *sum_emb = malloc(k * sizeof(float));
    float *gathered = malloc(d * k * sizeof(float));  // flattened (d, k)

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Initialize logits with bias
        for (int c = 0; c < C; c++) logits[c] = bias[c];

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        // Interaction part per class
        for (int c = 0; c < C; c++) {
            // Gather embeddings for this class and sample
            const float *emb_c = embed + ((long)c * d * 256 * k);
            for (int j = 0; j < d; j++) {
                const float *e = emb_c + ((long)j * 256 + xi[j]) * k;
                memcpy(gathered + j * k, e, k * sizeof(float));
            }
            // FM trick: 0.5 * ( (sum emb)^2 - sum(emb^2) )
            memset(sum_emb, 0, k * sizeof(float));
            float sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                for (int u = 0; u < k; u++) {
                    float v = gathered[j * k + u];
                    sum_emb[u] += v;
                    sum_sq += v * v;
                }
            }
            float sq_sum = 0.0f;
            for (int u = 0; u < k; u++) sq_sum += sum_emb[u] * sum_emb[u];
            logits[c] += 0.5f * (sq_sum - sum_sq);
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(logits); free(sum_emb); free(gathered);
    if (x) free(x);
    free(hd);
    return 0;
}
```