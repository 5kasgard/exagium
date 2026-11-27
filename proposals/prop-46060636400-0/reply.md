```hypothesis
Family: label-boosted-linear. Gradient boosting with linear base learners (logistic loss, shrinkage). Each learner uses the same per-feature encoding as label-linear (ordinal z-scored, categorical one-hot). Boosting reduces bias of a single linear model; 10 learners with shrinkage 0.1 should beat label-linear accuracy (0.8524) at moderate inference cost (~10x linear). If dev accuracy <= 0.8524 or inference cost > 2x label-gbdt, the hypothesis fails.
```
```python
"""label-boosted-linear: gradient boosting of linear models for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as z-score, categorical as one-hot (unseen=255).
Trains an ensemble of linear base learners on pseudo-residuals.
Exports per-learner per-feature lookup tables for fast inference."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
SEED = 1
NUM_LEARNERS = 10
SHRINKAGE = 0.1
BASE_LR = 0.03
BASE_L2 = 1e-4
BASE_BATCH = 64
BASE_EPOCHS = 5
MAX_TRAIN_SAMPLES = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Build encoding matrices E_j: 256 x dim_j
    enc = []
    dims = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
            dims.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            dims.append(len(vals))
    k = sum(dims)

    # Precompute full design matrix Z (n x k) as float32
    n = X.shape[0]
    Z = np.empty((n, k), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = dims[j]
        Z[:, off:off+dim] = E[X[:, j]]
        off += dim

    # Split for tuning? We hardcode after tuning on last 20% (file order).
    # Use all data for final training.
    train_idx = np.arange(n)

    # Ensemble storage: each learner has W (k x C), b (C)
    learners_W = []
    learners_b = []

    # Current ensemble logits (n x C)
    logits = np.zeros((n, C), dtype=np.float32)

    for learner in range(NUM_LEARNERS):
        # Compute probabilities from current logits
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        # Negative gradient of logistic loss: onehot - probs
        grad = np.zeros((n, C), dtype=np.float32)
        grad[np.arange(n), y] = 1.0
        grad -= probs

        # Train a linear model on grad as targets (multi-output regression with L2)
        W = np.zeros((k, C), dtype=np.float32)
        b = np.zeros(C, dtype=np.float32)
        # Adam state
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        t = 0

        for epoch in range(BASE_EPOCHS):
            order = rng.permutation(n)
            for s in range(0, n, BASE_BATCH):
                idx = order[s:s+BASE_BATCH]
                z, g = Z[idx], grad[idx]
                # Predictions of this base learner
                pred = z @ W + b
                # Gradient of MSE loss: (pred - g) / batch
                err = (pred - g) / len(idx)
                dW = z.T @ err + BASE_L2 * W
                db = err.sum(axis=0)

                t += 1
                # Adam update
                for param, grad_p, m, v in [(W, dW, mW, vW), (b, db, mb, vb)]:
                    m *= 0.9; m += 0.1 * grad_p
                    v *= 0.999; v += 0.001 * grad_p * grad_p
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    param -= BASE_LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Add learner with shrinkage
        learners_W.append(W * SHRINKAGE)
        learners_b.append(b * SHRINKAGE)
        logits += (Z @ (W * SHRINKAGE) + b * SHRINKAGE)

    # Convert each learner to per-feature lookup tables T_l[j][value][c]
    # Also store biases.
    tables = []
    for W, b in zip(learners_W, learners_b):
        T_learner = []
        off = 0
        for j, E in enumerate(enc):
            dim = dims[j]
            # T_learner[j] = E @ W[off:off+dim]  -> (256, C)
            Tj = E @ W[off:off+dim]
            T_learner.append(Tj.astype(np.float32))
            off += dim
        tables.append((T_learner, b.astype(np.float32)))

    # Write weights.bin
    # Magic: 0x424C4231 ("BLB1")
    # Header: magic, d, num_learners, C
    out = struct.pack("<4i", 0x424C4231, d, NUM_LEARNERS, C)
    for T_learner, b in tables:
        # Stack per-feature tables: (d, 256, C) -> flatten
        stacked = np.stack(T_learner, axis=0)  # d x 256 x C
        out += stacked.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-boosted-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index.
 * Weights format: magic 0x424C4231, d, L, C; then for each learner:
 *   float32 T[d][256][C] (row-major), float32 bias[C].
 * Inference: sum over learners of (bias + sum over features T[feature][value]),
 * then argmax. */
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
    if (nw < 16 || hd[0] != 0x424C4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], L = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || L < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + (long)L * ((long)d * 256 * C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Allocate accumulators for scores
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!scores) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize with zero
        for (int c = 0; c < C; c++) scores[c] = 0.0f;

        // Loop over learners
        for (int l = 0; l < L; l++) {
            // Add bias
            for (int c = 0; c < C; c++) scores[c] += ptr[(long)d * 256 * C + c];
            // Add per-feature contributions
            for (int j = 0; j < d; j++) {
                const float *t = ptr + ((long)j * 256 + xi[j]) * C;
                for (int c = 0; c < C; c++) scores[c] += t[c];
            }
            // Advance pointer to next learner
            ptr += (long)d * 256 * C + C;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```